"""Generates solution_v2.ipynb — the clean, frozen, generalisation-safe upgrade of the 0.44 stack.

Same pipeline as solution.ipynb, with the ONE proven change that lowered OOF 0.556 -> ~0.538:
  * SSL embeddings: mean over ALL hidden layers + mean&std time-pooling (was last-layer mean)
  * SSL PCA 48 -> 64
  * a few disfluency / syntactic-complexity linguistic features
Everything frozen (no fine-tuning). LGB+Ridge+SVR x 3 seeds, simplex on OOF RMSE, clip[0,5],
keyed to test. Judge by OOF (more seeds = more stable). solution.ipynb stays the safe 0.44.
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# SHL 2026 — Grammar Scoring v2 (frozen, generalisation-safe)
Your 0.44 stack + the one frozen change that measurably lowered OOF (0.556 -> ~0.538):
multi-layer SSL pooling (mean over all layers, mean&std over time), PCA-64, and a few
disfluency features. No fine-tuning. Decide by **OOF RMSE**; bump SEEDS for a stabler estimate.
""")

md("## 1 · Setup")
code("""import subprocess, sys, importlib
def pip(pkgs):
    for p in pkgs:
        try:
            subprocess.run([sys.executable,'-m','pip','install','-q','--only-binary=:all:',p],
                           check=True, timeout=240); print('ok  :', p)
        except Exception as e: print('SKIP:', p, '->', type(e).__name__)
pip(['lightgbm', 'language-tool-python'])
import os, gc, json, warnings, math, re, random
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, librosa, torch
from pathlib import Path
from tqdm.auto import tqdm
SEEDS=[42,1,2025]; N_FOLDS=5; SSL_PCA_DIM=64
SEED=SEEDS[0]; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
HAS_PRAAT=importlib.util.find_spec('parselmouth') is not None
HAS_LT   =importlib.util.find_spec('language_tool_python') is not None
print('device:',DEVICE,' praat:',HAS_PRAAT,' langtool:',HAS_LT)""")

md("## 2 · Load data")
code("""CANDIDATES=["/kaggle/input/shl-hiring-assessment-2026","/kaggle/input/shl-hiring-assessment-2026/Dataset_Final"]
DATA=next((Path(p) for p in CANDIDATES if Path(p).exists()),None)
if DATA is None:
    import kagglehub; DATA=Path(kagglehub.competition_download('shl-hiring-assessment-2026'))
def find(rgx):
    for p in DATA.rglob('*'):
        if re.search(rgx,p.name,re.I): return p
train_csv=find(r'^train\\.csv$'); test_csv=find(r'^test\\.csv$')
train_wav_dir=next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_train' in p.name.lower()),None) \\
           or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('train','train_audios','audio_train')),None)
test_wav_dir =next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_test' in p.name.lower()),None) \\
           or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('test','test_audios','audio_test')),None)
train=pd.read_csv(train_csv); test=pd.read_csv(test_csv)
LABEL_COL='label' if 'label' in train.columns else [c for c in train.columns if c!='filename'][0]
FILE_COL ='filename' if 'filename' in train.columns else train.columns[0]
print(train.shape,test.shape,'|',train_wav_dir,test_wav_dir)""")

md("## 3 · Acoustic features")
code("""if HAS_PRAAT:
    import parselmouth
    from parselmouth.praat import call
SR=16000
def acoustic_feats(path):
    y,sr=librosa.load(path,sr=SR,mono=True); dur=len(y)/sr
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    d1=librosa.feature.delta(mfcc); d2=librosa.feature.delta(mfcc,order=2)
    feats={}
    for name,arr in [('mfcc',mfcc),('d1',d1),('d2',d2)]:
        feats[f'{name}_mean']=arr.mean(); feats[f'{name}_std']=arr.std()
        for i in range(arr.shape[0]): feats[f'{name}{i}_m']=arr[i].mean(); feats[f'{name}{i}_s']=arr[i].std()
    for n,a in [('sc',librosa.feature.spectral_centroid(y=y,sr=sr)[0]),
                ('sbw',librosa.feature.spectral_bandwidth(y=y,sr=sr)[0]),
                ('sro',librosa.feature.spectral_rolloff(y=y,sr=sr)[0]),
                ('zcr',librosa.feature.zero_crossing_rate(y)[0]),
                ('rms',librosa.feature.rms(y=y)[0])]:
        feats[f'{n}_m']=a.mean(); feats[f'{n}_s']=a.std()
    iv=librosa.effects.split(y,top_db=30); speech=sum((b-a) for a,b in iv)/sr
    feats.update(dur=dur,speech_ratio=speech/dur,n_pauses=len(iv)-1,
                 mean_pause=np.mean([iv[i+1,0]-iv[i,1] for i in range(len(iv)-1)])/sr if len(iv)>1 else 0)
    try:
        assert HAS_PRAAT
        snd=parselmouth.Sound(y,sampling_frequency=sr); pv=snd.to_pitch().selected_array['frequency']; pv=pv[pv>0]
        feats['f0_m']=pv.mean() if len(pv) else 0; feats['f0_s']=pv.std() if len(pv) else 0
        pp=call(snd,"To PointProcess (periodic, cc)",75,500)
        feats['jitter']=call(pp,"Get jitter (local)",0,0,1e-4,0.02,1.3)
        feats['shimmer']=call([snd,pp],"Get shimmer (local)",0,0,1e-4,0.02,1.3,1.6)
    except Exception: feats.update(f0_m=0,f0_s=0,jitter=0,shimmer=0)
    return feats
def extract_all(df,wav_dir,tag):
    rows=[acoustic_feats(str(wav_dir/f)) for f in tqdm(df[FILE_COL],desc=f'ac-{tag}')]
    out=pd.DataFrame(rows); out.insert(0,FILE_COL,df[FILE_COL].values); return out
ac_tr=extract_all(train,train_wav_dir,'train'); ac_te=extract_all(test,test_wav_dir,'test')
print(ac_tr.shape)""")

md("## 4 · Whisper ASR + linguistic features (with disfluency/complexity)")
code("""from transformers import pipeline
asr=pipeline('automatic-speech-recognition',model='openai/whisper-small',
             device=0 if DEVICE=='cuda' else -1,chunk_length_s=30,stride_length_s=(5,5),
             torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32,
             generate_kwargs={'language':'en','task':'transcribe'})
def asr_df(df,wav_dir,tag):
    t=[]
    for f in tqdm(df[FILE_COL],desc=f'asr-{tag}'):
        try: t.append(asr(str(wav_dir/f))['text'].strip())
        except Exception as e: t.append(''); print('asr fail',f,e)
    return pd.DataFrame({FILE_COL:df[FILE_COL].values,'text':t})
tx_tr=asr_df(train,train_wav_dir,'train'); tx_te=asr_df(test,test_wav_dir,'test')
del asr; gc.collect(); torch.cuda.empty_cache()

import nltk; [nltk.download(x,quiet=True) for x in ('punkt','averaged_perceptron_tagger','punkt_tab','averaged_perceptron_tagger_eng')]
lt=None
if HAS_LT:
    import language_tool_python
    try: lt=language_tool_python.LanguageTool('en-US')
    except Exception as e: print('LT init failed:',e); lt=None
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
gpt_tok=GPT2TokenizerFast.from_pretrained('gpt2'); gpt=GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE).eval()
@torch.no_grad()
def perplexity(text,stride=512,max_len=1024):
    if not text.strip(): return 200.0
    enc=gpt_tok(text,return_tensors='pt').input_ids.to(DEVICE); nll,n=0.0,0
    for i in range(0,enc.size(1),stride):
        ch=enc[:,i:i+max_len]
        if ch.size(1)<2: break
        nll+=gpt(ch,labels=ch).loss.item()*(ch.size(1)-1); n+=ch.size(1)-1
    return math.exp(nll/max(n,1))
from collections import Counter
FILLERS={'um','uh','erm','er','hmm','mm','mmm','umm','uhh','ah','eh'}
def _mattr(w,win=50):
    if len(w)<=win: return len(set(w))/max(1,len(w))
    return float(np.mean([len(set(w[i:i+win]))/win for i in range(len(w)-win+1)]))
def ling_feats(text):
    text=text or ""; low=text.lower(); words=re.findall(r"\\w+",low)
    sents=nltk.sent_tokenize(text) if text else []
    n_w,n_s=max(1,len(words)),max(1,len(sents))
    errs=lt.check(text) if (text and lt is not None) else []
    pc=Counter(t for _,t in (nltk.pos_tag(words) if words else []))
    slens=[len(re.findall(r"\\w+",s.lower())) for s in sents] or [n_w]
    reps=sum(1 for i in range(1,len(words)) if words[i]==words[i-1])
    return {'n_words':n_w,'n_sents':n_s,'avg_wlen':np.mean([len(w) for w in words]) if words else 0,
            'avg_slen':n_w/n_s,'ttr':len(set(words))/n_w,'err_rate':len(errs)/n_w,'n_errs':len(errs),
            'ppl':perplexity(text[:4000]),'log_ppl':math.log1p(perplexity(text[:4000])),
            **{f'pos_{k}':pc.get(k,0)/n_w for k in ['NN','VB','JJ','RB','PRP','DT','IN','CC','MD']},
            'filler_rate':sum(low.count(f' {f} ') for f in FILLERS)/n_w,
            'immed_repeat_rate':reps/n_w,'mattr50':_mattr(words),
            'longword_rate':sum(1 for w in words if len(w)>=7)/n_w,'comma_rate':text.count(',')/n_w,
            'slen_std':float(np.std(slens)),'subordinate_rate':pc.get('IN',0)/n_w,
            'content_ratio':sum(pc.get(k,0) for k in ['NN','VB','JJ','RB'])/n_w}
def build_ling(tx,tag):
    rows=[ling_feats(t) for t in tqdm(tx['text'],desc=f'ling-{tag}')]
    out=pd.DataFrame(rows); out.insert(0,FILE_COL,tx[FILE_COL].values); return out
lg_tr=build_ling(tx_tr,'train'); lg_te=build_ling(tx_te,'test')
del gpt,gpt_tok; gc.collect(); torch.cuda.empty_cache()
print(lg_tr.shape)""")

md("## 5 · SSL embeddings — multi-layer mean + mean&std pooling")
code("""from transformers import AutoFeatureExtractor, AutoModel
MIN_SAMPLES=16000
def load_ssl(name):
    return AutoFeatureExtractor.from_pretrained(name), AutoModel.from_pretrained(name,output_hidden_states=True).to(DEVICE).eval()
@torch.no_grad()
def ssl_embed(path,fe,mdl,chunk_s=20):
    try:
        y,_=librosa.load(path,sr=16000,mono=True)
        if len(y)<MIN_SAMPLES: y=np.pad(y,(0,MIN_SAMPLES-len(y)))
        step=chunk_s*16000
        chunks=[c for c in (y[i:i+step] for i in range(0,len(y),step)) if len(c)>=MIN_SAMPLES] \\
               or [np.pad(y,(0,max(0,MIN_SAMPLES-len(y))))]
        vecs=[]
        for c in chunks:
            inp=fe(c,sampling_rate=16000,return_tensors='pt').input_values.to(DEVICE)
            st=torch.stack(mdl(inp).hidden_states,0)     # [L,1,T,H]
            m=st.mean(2).mean(0).squeeze(0)              # mean over time then layers
            s=st.std(2).mean(0).squeeze(0)               # std over time then mean layers
            vecs.append(torch.cat([m,s]).float().cpu().numpy())
        return np.mean(vecs,0)
    except Exception as e:
        print('ssl fail',path,e); return None
def ssl_df(df,wav_dir,tag,prefix,fe,mdl):
    rows=[ssl_embed(str(wav_dir/f),fe,mdl) for f in tqdm(df[FILE_COL],desc=f'{prefix}-{tag}')]
    dim=max((len(r) for r in rows if r is not None),default=1)
    rows=[r if r is not None else np.zeros(dim,np.float32) for r in rows]
    E=np.vstack(rows).astype(np.float32)
    out=pd.DataFrame(E,columns=[f'{prefix}_raw{i}' for i in range(E.shape[1])])
    out.insert(0,FILE_COL,df[FILE_COL].values); return out
fe,mdl=load_ssl('facebook/wav2vec2-large-960h')
w2_tr=ssl_df(train,train_wav_dir,'train','w2v',fe,mdl); w2_te=ssl_df(test,test_wav_dir,'test','w2v',fe,mdl)
del mdl,fe; gc.collect(); torch.cuda.empty_cache()
fe,mdl=load_ssl('microsoft/wavlm-large')
wl_tr=ssl_df(train,train_wav_dir,'train','wl',fe,mdl); wl_te=ssl_df(test,test_wav_dir,'test','wl',fe,mdl)
del mdl,fe; gc.collect(); torch.cuda.empty_cache()
print('w2v',w2_tr.shape,'wavlm',wl_tr.shape)""")

md("## 6 · Assemble (standardise -> PCA-64 on each SSL block)")
code("""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functools import reduce
def pca_block(tr_raw,te_raw,prefix,dim=SSL_PCA_DIM):
    cols=[c for c in tr_raw.columns if c!=FILE_COL]
    Xtr=np.nan_to_num(tr_raw[cols].values.astype(np.float32)); Xte=np.nan_to_num(te_raw[cols].values.astype(np.float32))
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    k=int(min(dim,Xtr.shape[1],Xtr.shape[0]-1))
    p=PCA(n_components=k,random_state=SEED).fit(Xtr)
    tr=pd.DataFrame(p.transform(Xtr),columns=[f'{prefix}_pc{i}' for i in range(k)]); tr.insert(0,FILE_COL,tr_raw[FILE_COL].values)
    te=pd.DataFrame(p.transform(Xte),columns=[f'{prefix}_pc{i}' for i in range(k)]); te.insert(0,FILE_COL,te_raw[FILE_COL].values)
    print(f'{prefix}: {Xtr.shape[1]}->{k} EVR={p.explained_variance_ratio_.sum():.3f}'); return tr,te
w2p_tr,w2p_te=pca_block(w2_tr,w2_te,'w2v'); wlp_tr,wlp_te=pca_block(wl_tr,wl_te,'wl')
X_tr_full=reduce(lambda a,b:a.merge(b,on=FILE_COL),[ac_tr,lg_tr,w2p_tr,wlp_tr])
X_te_full=reduce(lambda a,b:a.merge(b,on=FILE_COL),[ac_te,lg_te,w2p_te,wlp_te])
y=train.merge(X_tr_full[[FILE_COL]],on=FILE_COL)[LABEL_COL].astype(float).values
FEATS=[c for c in X_tr_full.columns if c!=FILE_COL]
X_tr=X_tr_full[FEATS].astype(np.float32).replace([np.inf,-np.inf],0).fillna(0).values
X_te=X_te_full[FEATS].astype(np.float32).replace([np.inf,-np.inf],0).fillna(0).values
print('X',X_tr.shape,X_te.shape)""")

md("## 7 · CV blend — LightGBM + Ridge + SVR (3 seeds, simplex on OOF RMSE)")
code("""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
bins=pd.cut(y,bins=[-0.1,1.5,2.5,3.5,4.5,5.1],labels=False)
lgb_params=dict(objective='regression',metric='rmse',learning_rate=0.03,num_leaves=31,
                feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=8,lambda_l2=1.0,verbose=-1)
oof={m:np.zeros(len(y)) for m in ['lgb','rdg','svr']}; pred={m:np.zeros(len(X_te)) for m in ['lgb','rdg','svr']}
for seed in SEEDS:
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    for tr,va in skf.split(X_tr,bins):
        m=lgb.train({**lgb_params,'seed':seed},lgb.Dataset(X_tr[tr],y[tr]),num_boost_round=4000,
                    valid_sets=[lgb.Dataset(X_tr[va],y[va])],callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        oof['lgb'][va]+=m.predict(X_tr[va])/len(SEEDS); pred['lgb']+=m.predict(X_te)/(N_FOLDS*len(SEEDS))
        sc=StandardScaler().fit(X_tr[tr]); Ztr,Zva,Zte=sc.transform(X_tr[tr]),sc.transform(X_tr[va]),sc.transform(X_te)
        r=Ridge(alpha=8.0).fit(Ztr,y[tr]); oof['rdg'][va]+=r.predict(Zva)/len(SEEDS); pred['rdg']+=r.predict(Zte)/(N_FOLDS*len(SEEDS))
        s=SVR(kernel='rbf',C=10,epsilon=0.2).fit(Ztr,y[tr]); oof['svr'][va]+=s.predict(Zva)/len(SEEDS); pred['svr']+=s.predict(Zte)/(N_FOLDS*len(SEEDS))
for m in ['lgb','rdg','svr']: print(f'{m} OOF RMSE={rmse(y,oof[m]):.4f}  Pearson={pearsonr(y,oof[m])[0]:.4f}')
g=np.linspace(0,1,21); best=None
for a in g:
    for b in g:
        if a+b>1: continue
        e=rmse(y,a*oof['lgb']+b*oof['rdg']+(1-a-b)*oof['svr'])
        if best is None or e<best[0]: best=(e,(a,b,1-a-b))
w=best[1]
oof_blend=w[0]*oof['lgb']+w[1]*oof['rdg']+w[2]*oof['svr']
preds=np.clip(w[0]*pred['lgb']+w[1]*pred['rdg']+w[2]*pred['svr'],0,5)
print(f'blend w(lgb,rdg,svr)={tuple(round(x,3) for x in w)}')
print(f'>> OOF RMSE={rmse(y,oof_blend):.4f}  Pearson={pearsonr(y,oof_blend)[0]:.4f}  (0.44 stack was 0.556)')""")

md("## 8 · Submission (keyed to test, clip [0,5])")
code("""sub=pd.DataFrame({FILE_COL:test[FILE_COL].values})
sub['label']=sub[FILE_COL].map(dict(zip(X_te_full[FILE_COL].values,preds))).fillna(float(np.mean(y)))
sub['label']=np.clip(sub['label'],0,5)
sub.to_csv('submission.csv',index=False)
assert len(sub)==len(test) and sub['label'].notna().all()
print(sub.head()); print('rows:',len(sub),'range:',round(sub.label.min(),3),'..',round(sub.label.max(),3))""")

nb['cells']=cells
nbf.write(nb,'solution_v2.ipynb')
print('wrote solution_v2.ipynb with',len(cells),'cells')

"""Generates solution_v3.ipynb — v2 features + a better-engineered, more-honest pipeline.

Adds vs solution_v2:
  1. Feature caching  — transcripts / acoustic / SSL cached to disk; re-runs skip recompute.
  2. In-fold PCA + scaling — fit ONLY on each fold's train (no leakage) -> trustworthy OOF.
  3. OOF stability report — per-fold RMSE mean +/- std, so you see if a gain generalises.
  4. ExtraTrees added to the blend (decorrelated from LGB) + NNLS weight fit (robust).
Frozen, no fine-tuning. solution.ipynb (0.44) and solution_v2.ipynb stay untouched.
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# SHL 2026 — Grammar Scoring v3 (hardened pipeline)
Same frozen features as v2, but the pipeline is engineered better:
**cached features** (fast re-runs), **in-fold PCA/scaling** (leakage-free -> OOF you can trust
for the private set), an **OOF stability read** (per-fold mean +/- std), and a **4th model
(ExtraTrees)** blended with **NNLS**. Decide on OOF; the std tells you if it's stable.
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
# ---- feature cache: skip recompute if the file already exists (huge time save on re-runs) ----
def cache_pq(name, build):
    if Path(name).exists(): print('cache hit :', name); return pd.read_parquet(name)
    df=build(); df.to_parquet(name); print('cached    :', name); return df
def cache_csv(name, build):
    if Path(name).exists(): print('cache hit :', name); return pd.read_csv(name)
    df=build(); df.to_csv(name,index=False); print('cached    :', name); return df
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

md("## 3 · Acoustic features (cached)")
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
def build_ac(df,wav_dir,tag):
    def _b():
        rows=[acoustic_feats(str(wav_dir/f)) for f in tqdm(df[FILE_COL],desc=f'ac-{tag}')]
        out=pd.DataFrame(rows); out.insert(0,FILE_COL,df[FILE_COL].values); return out
    return cache_pq(f'ac_{tag}.parquet',_b)
ac_tr=build_ac(train,train_wav_dir,'tr'); ac_te=build_ac(test,test_wav_dir,'te')
print(ac_tr.shape)""")

md("## 4 · Whisper ASR (cached) + linguistic features (cached)")
code("""def build_asr():
    from transformers import pipeline
    asr=pipeline('automatic-speech-recognition',model='openai/whisper-small',
                 device=0 if DEVICE=='cuda' else -1,chunk_length_s=30,stride_length_s=(5,5),
                 torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32,
                 generate_kwargs={'language':'en','task':'transcribe'})
    def _one(df,wav_dir,tag):
        t=[]
        for f in tqdm(df[FILE_COL],desc=f'asr-{tag}'):
            try: t.append(asr(str(wav_dir/f))['text'].strip())
            except Exception as e: t.append(''); print('asr fail',f,e)
        return pd.DataFrame({FILE_COL:df[FILE_COL].values,'text':t})
    tr=_one(train,train_wav_dir,'tr'); te=_one(test,test_wav_dir,'te')
    del asr; gc.collect(); torch.cuda.empty_cache(); return tr,te
if Path('tx_tr.csv').exists() and Path('tx_te.csv').exists():
    tx_tr=pd.read_csv('tx_tr.csv'); tx_te=pd.read_csv('tx_te.csv'); print('cache hit : transcripts')
    tx_tr['text']=tx_tr['text'].fillna(''); tx_te['text']=tx_te['text'].fillna('')
else:
    tx_tr,tx_te=build_asr(); tx_tr.to_csv('tx_tr.csv',index=False); tx_te.to_csv('tx_te.csv',index=False)

import nltk; [nltk.download(x,quiet=True) for x in ('punkt','averaged_perceptron_tagger','punkt_tab','averaged_perceptron_tagger_eng')]
lt=None
if HAS_LT:
    import language_tool_python
    try: lt=language_tool_python.LanguageTool('en-US')
    except Exception as e: print('LT init failed:',e); lt=None
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import Counter
FILLERS={'um','uh','erm','er','hmm','mm','mmm','umm','uhh','ah','eh'}
def _mattr(w,win=50):
    if len(w)<=win: return len(set(w))/max(1,len(w))
    return float(np.mean([len(set(w[i:i+win]))/win for i in range(len(w)-win+1)]))
def build_ling(tx,tag):
    def _b():
        gpt_tok=GPT2TokenizerFast.from_pretrained('gpt2'); gpt=GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE).eval()
        @torch.no_grad()
        def ppl(text,stride=512,max_len=1024):
            if not text.strip(): return 200.0
            enc=gpt_tok(text,return_tensors='pt').input_ids.to(DEVICE); nll,n=0.0,0
            for i in range(0,enc.size(1),stride):
                ch=enc[:,i:i+max_len]
                if ch.size(1)<2: break
                nll+=gpt(ch,labels=ch).loss.item()*(ch.size(1)-1); n+=ch.size(1)-1
            return math.exp(nll/max(n,1))
        def feats(text):
            text=text or ""; low=text.lower(); words=re.findall(r"\\w+",low)
            sents=nltk.sent_tokenize(text) if text else []
            n_w,n_s=max(1,len(words)),max(1,len(sents))
            errs=lt.check(text) if (text and lt is not None) else []
            pc=Counter(t for _,t in (nltk.pos_tag(words) if words else []))
            slens=[len(re.findall(r"\\w+",s.lower())) for s in sents] or [n_w]
            reps=sum(1 for i in range(1,len(words)) if words[i]==words[i-1])
            return {'n_words':n_w,'n_sents':n_s,'avg_wlen':np.mean([len(w) for w in words]) if words else 0,
                    'avg_slen':n_w/n_s,'ttr':len(set(words))/n_w,'err_rate':len(errs)/n_w,'n_errs':len(errs),
                    'ppl':ppl(text[:4000]),'log_ppl':math.log1p(ppl(text[:4000])),
                    **{f'pos_{k}':pc.get(k,0)/n_w for k in ['NN','VB','JJ','RB','PRP','DT','IN','CC','MD']},
                    'filler_rate':sum(low.count(f' {f} ') for f in FILLERS)/n_w,
                    'immed_repeat_rate':reps/n_w,'mattr50':_mattr(words),
                    'longword_rate':sum(1 for w in words if len(w)>=7)/n_w,'comma_rate':text.count(',')/n_w,
                    'slen_std':float(np.std(slens)),'subordinate_rate':pc.get('IN',0)/n_w,
                    'content_ratio':sum(pc.get(k,0) for k in ['NN','VB','JJ','RB'])/n_w}
        rows=[feats(t) for t in tqdm(tx['text'],desc=f'ling-{tag}')]
        del gpt,gpt_tok; gc.collect(); torch.cuda.empty_cache()
        out=pd.DataFrame(rows); out.insert(0,FILE_COL,tx[FILE_COL].values); return out
    return cache_pq(f'lg_{tag}.parquet',_b)
lg_tr=build_ling(tx_tr,'tr'); lg_te=build_ling(tx_te,'te')
print(lg_tr.shape)""")

md("## 5 · SSL raw embeddings (multi-layer mean+std, cached — PCA happens in-fold later)")
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
            st=torch.stack(mdl(inp).hidden_states,0)
            m=st.mean(2).mean(0).squeeze(0); s=st.std(2).mean(0).squeeze(0)
            vecs.append(torch.cat([m,s]).float().cpu().numpy())
        return np.mean(vecs,0)
    except Exception as e:
        print('ssl fail',path,e); return None
def build_ssl(model_name, prefix):
    def _one(df,wav_dir,tag,fe,mdl):
        rows=[ssl_embed(str(wav_dir/f),fe,mdl) for f in tqdm(df[FILE_COL],desc=f'{prefix}-{tag}')]
        dim=max((len(r) for r in rows if r is not None),default=1)
        rows=[r if r is not None else np.zeros(dim,np.float32) for r in rows]
        E=np.vstack(rows).astype(np.float32)
        out=pd.DataFrame(E,columns=[f'{prefix}_raw{i}' for i in range(E.shape[1])])
        out.insert(0,FILE_COL,df[FILE_COL].values); return out
    if Path(f'{prefix}_tr.parquet').exists() and Path(f'{prefix}_te.parquet').exists():
        print('cache hit :',prefix); return pd.read_parquet(f'{prefix}_tr.parquet'), pd.read_parquet(f'{prefix}_te.parquet')
    fe,mdl=load_ssl(model_name)
    tr=_one(train,train_wav_dir,'tr',fe,mdl); te=_one(test,test_wav_dir,'te',fe,mdl)
    del mdl,fe; gc.collect(); torch.cuda.empty_cache()
    tr.to_parquet(f'{prefix}_tr.parquet'); te.to_parquet(f'{prefix}_te.parquet'); return tr,te
w2_tr,w2_te=build_ssl('facebook/wav2vec2-large-960h','w2v')
wl_tr,wl_te=build_ssl('microsoft/wavlm-large','wl')
print('w2v',w2_tr.shape,'wavlm',wl_tr.shape)""")

md("## 6 · Model — in-fold PCA/scaling, 4 models, NNLS blend, stability report")
code("""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from functools import reduce
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))

# assemble aligned raw blocks
Mtr=reduce(lambda a,b:a.merge(b,on=FILE_COL),[ac_tr,lg_tr,w2_tr,wl_tr])
Mte=reduce(lambda a,b:a.merge(b,on=FILE_COL),[ac_te,lg_te,w2_te,wl_te])
y=train.merge(Mtr[[FILE_COL]],on=FILE_COL)[LABEL_COL].astype(float).values
w2c=[c for c in Mtr.columns if c.startswith('w2v_raw')]; wlc=[c for c in Mtr.columns if c.startswith('wl_raw')]
handc=[c for c in Mtr.columns if c!=FILE_COL and c not in w2c and c not in wlc]
def _v(df,cols): return np.nan_to_num(df[cols].values.astype(np.float32))
Atr,Ate=_v(Mtr,handc),_v(Mte,handc)      # handcrafted (no PCA)
Wtr,Wte=_v(Mtr,w2c),_v(Mte,w2c)          # wav2vec2 raw (PCA in-fold)
Ltr,Lte=_v(Mtr,wlc),_v(Mte,wlc)          # wavlm raw   (PCA in-fold)
bins=pd.cut(y,bins=[-0.1,1.5,2.5,3.5,4.5,5.1],labels=False)
print('handcrafted',Atr.shape[1],'| w2v raw',Wtr.shape[1],'| wavlm raw',Ltr.shape[1])

def infold_pca(Rtr,Rte,tr,dim,seed):
    sc=StandardScaler().fit(Rtr[tr]); ztr=sc.transform(Rtr); zte=sc.transform(Rte)
    k=int(min(dim,ztr.shape[1],len(tr)-1))
    p=PCA(n_components=k,random_state=seed).fit(ztr[tr])
    return p.transform(ztr), p.transform(zte)

lgb_params=dict(objective='regression',metric='rmse',learning_rate=0.03,num_leaves=31,
                feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=8,lambda_l2=1.0,verbose=-1)
MODELS=['lgb','rdg','svr','et']
oof={m:np.zeros(len(y)) for m in MODELS}; pred={m:np.zeros(len(Ate)) for m in MODELS}
for seed in SEEDS:
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    for tr,va in skf.split(np.zeros(len(y)),bins):
        w2p_tr,w2p_te=infold_pca(Wtr,Wte,tr,SSL_PCA_DIM,seed)     # leakage-free: fit on train only
        wlp_tr,wlp_te=infold_pca(Ltr,Lte,tr,SSL_PCA_DIM,seed)
        Xtr=np.hstack([Atr,w2p_tr,wlp_tr]); Xte=np.hstack([Ate,w2p_te,wlp_te])
        m=lgb.train({**lgb_params,'seed':seed},lgb.Dataset(Xtr[tr],y[tr]),num_boost_round=4000,
                    valid_sets=[lgb.Dataset(Xtr[va],y[va])],callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        oof['lgb'][va]+=m.predict(Xtr[va])/len(SEEDS); pred['lgb']+=m.predict(Xte)/(N_FOLDS*len(SEEDS))
        et=ExtraTreesRegressor(n_estimators=600,max_features=0.5,min_samples_leaf=3,n_jobs=-1,random_state=seed).fit(Xtr[tr],y[tr])
        oof['et'][va]+=et.predict(Xtr[va])/len(SEEDS); pred['et']+=et.predict(Xte)/(N_FOLDS*len(SEEDS))
        sc=StandardScaler().fit(Xtr[tr]); Ztr,Zva,Zte=sc.transform(Xtr[tr]),sc.transform(Xtr[va]),sc.transform(Xte)
        r=Ridge(alpha=8.0).fit(Ztr,y[tr]); oof['rdg'][va]+=r.predict(Zva)/len(SEEDS); pred['rdg']+=r.predict(Zte)/(N_FOLDS*len(SEEDS))
        s=SVR(kernel='rbf',C=10,epsilon=0.2).fit(Ztr,y[tr]); oof['svr'][va]+=s.predict(Zva)/len(SEEDS); pred['svr']+=s.predict(Zte)/(N_FOLDS*len(SEEDS))
for m in MODELS: print(f'{m} OOF RMSE={rmse(y,oof[m]):.4f}  Pearson={pearsonr(y,oof[m])[0]:.4f}')

# robust blend: non-negative least squares on OOF (minimises OOF RMSE, few params -> low overfit)
Moof=np.stack([oof[m] for m in MODELS],1); wv,_=nnls(Moof,y)
oof_blend=Moof@wv
preds=np.clip(np.stack([pred[m] for m in MODELS],1)@wv,0,5)
print('NNLS weights:',{m:round(float(x),3) for m,x in zip(MODELS,wv)})
print(f'>> OOF RMSE={rmse(y,oof_blend):.4f}  Pearson={pearsonr(y,oof_blend)[0]:.4f}  (v2 was ~0.538)')

# stability: spread of per-fold OOF RMSE (low std = generalises evenly)
fr=[]
for seed in SEEDS:
    for tr,va in StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed).split(np.zeros(len(y)),bins):
        fr.append(rmse(y[va],np.clip(oof_blend[va],0,5)))
print(f'per-fold OOF RMSE: mean={np.mean(fr):.4f}  std={np.std(fr):.4f}')""")

md("## 7 · Submission (keyed to test, clip [0,5])")
code("""sub=pd.DataFrame({FILE_COL:test[FILE_COL].values})
sub['label']=sub[FILE_COL].map(dict(zip(Mte[FILE_COL].values,preds))).fillna(float(np.mean(y)))
sub['label']=np.clip(sub['label'],0,5)
sub.to_csv('submission.csv',index=False)
assert len(sub)==len(test) and sub['label'].notna().all()
print(sub.head()); print('rows:',len(sub),'range:',round(sub.label.min(),3),'..',round(sub.label.max(),3))""")

nb['cells']=cells
nbf.write(nb,'solution_v3.ipynb')
print('wrote solution_v3.ipynb with',len(cells),'cells')

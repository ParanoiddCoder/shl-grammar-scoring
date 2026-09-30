"""Generates solution_v4.ipynb — v3's hardened pipeline + the research-backed grammar signals.

Rationale (grounded in the literature — see EXPERIMENTS.md v4 section):
  * The target is GRAMMAR, yet v3 spends ~95% of its feature budget on how speech *sounds*
    (MFCC/spectral + 4096 raw SSL dims) and almost none on whether it is *grammatical*.
    That is why v3 plateaus at OOF ~0.53.
  * "Back to grammar" (Speech Communication 2023) and essay-scoring MTL work show the strongest
    grammar signal is grammatical-error-correction (GEC) edit statistics on transcripts.
  * The 2026 L2-speaking paper (arXiv 2608.26137) reaches p=0.818 (beating the single-human
    ceiling) with an interpretable fluency composite BLENDED with a zero-shot LLM judge.
  * Every text feature rides on transcript quality -> upgrade ASR (whisper-large-v3-turbo via
    faster-whisper), which ALSO yields word-confidence + timing for free.

Adds vs v3 (each behind a CONFIG flag, each cached, each judged on in-fold OOF):
  1. ASR upgrade: faster-whisper large-v3-turbo -> transcripts + per-segment + per-word stats.
  2. De-Jong fluency features (mean length of run, pause ratio, long-pause rate, speech rate...).
  3. ASR-confidence features (avg_logprob / no_speech_prob / compression_ratio / word prob).
  4. GEC edit-rate + error-type features (flan-t5 grammar-synthesis, per-sentence, aggregated).
  5. Zero-shot LLM grammar judge (Qwen2.5-Instruct) -> feature AND an NNLS blend member.
  6. Honest nested-CV isotonic calibration (monotonic -> helps RMSE, leaves Pearson intact).
  7. Report + visualizations section (interpretability is graded by the brief).

Frozen features only, no end-to-end fine-tuning. v3 / v2 / 0.44 notebooks stay untouched.
Caches that are ASR-independent (acoustic ac_*, SSL w2v_*/wl_*) are REUSED from v3 runs;
everything transcript-dependent is written with a _v4 suffix so it never collides with v3.
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# SHL 2026 — Grammar Scoring v4 (grammar-signal pipeline)
Built on v3's leakage-free backbone (in-fold PCA/scaling, NNLS blend, per-fold stability), this
version adds the signals the literature shows actually measure **grammar** on spoken L2 English:

1. **Better ASR** — faster-whisper `large-v3-turbo` (cleaner transcripts lift every text feature).
2. **De-Jong fluency** — mean length of run, pause ratio, long-pause rate, speech/articulation rate.
3. **ASR confidence** — avg log-prob, no-speech-prob, compression ratio, word probability.
4. **GEC error features** — a grammar-error-correction model rewrites each sentence; we measure how
   much it had to change (edit rate + insertion/deletion/substitution rates + worst-sentence rate).
5. **Zero-shot LLM judge** — an instruction model rates grammar 0–5; used as a feature *and* as an
   NNLS blend member (the 2026 SOTA pattern: the judge refines, the composite breaks ties).
6. **Honest calibration** — nested-CV isotonic; monotonic so it improves RMSE without moving Pearson.

Every addition is a **frozen / pretrained** signal, scored on **in-fold OOF RMSE + per-fold std**.
Toggle any lever in the **CONFIG** cell and keep it only if OOF improves.
""")

md("## 0 · CONFIG — one lever per run; keep it only if in-fold OOF RMSE improves")
code("""# ---- data / CV (same backbone as v3) ----
SEEDS       = [42, 1, 2025]
N_FOLDS     = 5
SSL_PCA_DIM = 64

# ---- ASR ----
ASR_MODEL      = 'deepdml/faster-whisper-large-v3-turbo-ct2'  # fallback -> 'large-v3' -> 'small'
ASR_BEAM       = 5

# ---- new grammar / fluency signals ----
USE_DEJONG   = True      # timing/fluency features from word timestamps
USE_ASR_CONF = True      # decoder-confidence features (nearly free)
USE_GEC      = True      # grammatical-error-correction edit statistics  (biggest lever)
GEC_MODEL    = 'pszemraj/flan-t5-large-grammar-synthesis'  # robust, idempotent on clean text
GEC_PREFIX   = ''        # this model needs no prefix; for 'vennify/t5-base-grammar-correction' use 'grammar: '
GEC_MAX_SENTS= 14        # cap sentences/clip (runtime guard)
GEC_BATCH    = 16

USE_LLM_JUDGE = True     # zero-shot 0-5 grammar rating -> feature + blend member
LLM_MODEL     = 'Qwen/Qwen2.5-3B-Instruct'   # fp16 fits T4; '...-7B-Instruct' is a better judge if memory allows
LLM_AS_BLEND  = True     # also add the raw judge as an NNLS blend member (SOTA pattern)

# ---- calibration ----
APPLY_CALIBRATION = True # apply isotonic to final preds ONLY (nested delta is printed either way)

# ---- keep v3 features on ----
USE_SSL = True
print('CONFIG loaded')""")

md("## 1 · Setup")
code("""import subprocess, sys, importlib
def pip(pkgs):
    for p in pkgs:
        ok=False
        for extra in (['--only-binary=:all:'], []):
            try:
                subprocess.run([sys.executable,'-m','pip','install','-q',*extra,p],check=True,timeout=600)
                print('ok  :',p); ok=True; break
            except Exception as e: last=e
        if not ok: print('SKIP:',p,'->',type(last).__name__)
pip(['lightgbm','faster-whisper','sentencepiece','accelerate','language-tool-python'])
import os, gc, json, warnings, math, re, random, difflib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, librosa, torch
from pathlib import Path
from tqdm.auto import tqdm
SEED=SEEDS[0]; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
HAS_PRAAT=importlib.util.find_spec('parselmouth') is not None
HAS_LT   =importlib.util.find_spec('language_tool_python') is not None
def cache_pq(name, build):
    if Path(name).exists(): print('cache hit :',name); return pd.read_parquet(name)
    df=build(); df.to_parquet(name); print('cached    :',name); return df
print('device:',DEVICE,' praat:',HAS_PRAAT,' langtool:',HAS_LT)""")

md("## 2 · Load data (keyed to test.csv — never sample_submission)")
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

md("## 3 · Acoustic features (cached — reused from v3, ASR-independent)")
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

md("""## 4 · ASR — faster-whisper large-v3-turbo (cached) → transcripts + timing + confidence
One decode pass yields three things at once: the **transcript** (feeds §5/§6/§7), **word timestamps**
(feed De-Jong fluency), and **decoder confidence** (avg log-prob / no-speech-prob / compression ratio).
""")
code("""def build_asr_v4():
    from faster_whisper import WhisperModel
    def _load(name):
        return WhisperModel(name, device='cuda' if DEVICE=='cuda' else 'cpu',
                            compute_type='float16' if DEVICE=='cuda' else 'int8')
    model=None
    for cand in [ASR_MODEL,'large-v3','small']:
        try: model=_load(cand); print('ASR model loaded:',cand); break
        except Exception as e: print('ASR load FAILED',cand,'->',repr(e))
    if model is None:
        raise RuntimeError('faster-whisper could not load ANY model (turbo/large-v3/small). '
                           'Check GPU + Internet=ON on Kaggle; do not proceed with empty transcripts.')
    def _one(path):
        # decode with librosa (robust) and pass the ndarray -> bypasses faster-whisper's PyAV
        # audio loader, which clashes with Kaggle's `av` ('metadata_errors' TypeError).
        y,_=librosa.load(str(path),sr=16000,mono=True); y=y.astype(np.float32)
        segments,info=model.transcribe(y,language='en',beam_size=ASR_BEAM,
                                       word_timestamps=True,vad_filter=True)
        segs=[]; words=[]
        for s in segments:
            segs.append(dict(text=s.text,alp=float(s.avg_logprob),nsp=float(s.no_speech_prob),
                             cr=float(s.compression_ratio),start=float(s.start),end=float(s.end)))
            for w in (s.words or []):
                words.append([float(w.start),float(w.end),float(w.probability)])
        return dict(dur=float(info.duration),text=' '.join(x['text'].strip() for x in segs).strip(),
                    segs=segs,words=words)
    def _run(df,wav_dir,tag):
        meta={}; empt=0
        for f in tqdm(df[FILE_COL],desc=f'asr-{tag}'):
            try:
                r=_one(wav_dir/f); meta[f]=r
                if not r['text']: empt+=1
            except Exception as e: print('asr fail',f,repr(e)); meta[f]=dict(dur=0.0,text='',segs=[],words=[]); empt+=1
        print(f'asr-{tag}: {empt}/{len(df)} empty/failed transcripts')
        return meta
    tr=_run(train,train_wav_dir,'tr'); te=_run(test,test_wav_dir,'te')
    del model; gc.collect(); torch.cuda.empty_cache(); return tr,te

def _meanwords(meta,df): return float(np.mean([len((meta[f]['text'] or '').split()) for f in df[FILE_COL]]))
# only trust a cache that actually holds transcripts; a stale EMPTY cache is deleted and recomputed
if Path('asrmeta_tr_v4.json').exists() and Path('asrmeta_te_v4.json').exists():
    meta_tr=json.load(open('asrmeta_tr_v4.json')); meta_te=json.load(open('asrmeta_te_v4.json'))
    if _meanwords(meta_tr,train)==0:
        print('!! cached ASR meta is EMPTY -> deleting and recomputing'); os.remove('asrmeta_tr_v4.json'); os.remove('asrmeta_te_v4.json')
        meta_tr,meta_te=build_asr_v4()
    else: print('cache hit : asr meta v4')
else:
    meta_tr,meta_te=build_asr_v4()
# transcripts frame (compat with v3-style linguistic builder)
tx_tr=pd.DataFrame({FILE_COL:train[FILE_COL].values,'text':[meta_tr[f]['text'] for f in train[FILE_COL]]})
tx_te=pd.DataFrame({FILE_COL:test[FILE_COL].values ,'text':[meta_te[f]['text'] for f in test[FILE_COL]]})
tx_tr['text']=tx_tr['text'].fillna(''); tx_te['text']=tx_te['text'].fillna('')
mw=float(tx_tr['text'].str.split().apply(len).mean())
print('mean transcript words (train):',round(mw,1),'| sample:',repr(tx_tr['text'].iloc[0][:120]))
# only cache once transcripts are real; fail-fast so downstream grammar features are never built on empties
assert mw>0, ('ASR produced EMPTY transcripts -> GEC/LLM/fluency would be dead. Fix faster-whisper '
              '(see ASR load lines above) and re-run. Nothing cached.')
json.dump(meta_tr,open('asrmeta_tr_v4.json','w')); json.dump(meta_te,open('asrmeta_te_v4.json','w'))""")

md("### 4b · De-Jong fluency + ASR-confidence features (from the cached ASR meta)")
code("""PAUSE_MIN=0.25; PAUSE_LONG=0.60
def dejong_feats(m):
    dur=m.get('dur',0.0); ws=sorted(m.get('words',[]),key=lambda x:x[0])
    if dur<=0 or len(ws)==0:
        return dict(dj_nwords=0,dj_speech_rate=0,dj_artic_rate=0,dj_pause_ratio=0,dj_npauses=0,
                    dj_pause_permin=0,dj_longpause_permin=0,dj_mean_run=0,dj_std_run=0,dj_max_run=0,dj_mean_pause=0)
    gaps=[max(0.0,ws[i][0]-ws[i-1][1]) for i in range(1,len(ws))]
    paus=[g for g in gaps if g>=PAUSE_MIN]; longp=[g for g in gaps if g>=PAUSE_LONG]
    sil=float(sum(paus)); speech=max(1e-6,dur-sil); nw=len(ws); mins=max(1e-6,dur/60.0)
    runs=[]; cur=1
    for g in gaps:
        if g>=PAUSE_MIN: runs.append(cur); cur=1
        else: cur+=1
    runs.append(cur)
    return dict(dj_nwords=nw,dj_speech_rate=nw/dur,dj_artic_rate=nw/speech,dj_pause_ratio=sil/dur,
                dj_npauses=len(paus),dj_pause_permin=len(paus)/mins,dj_longpause_permin=len(longp)/mins,
                dj_mean_run=float(np.mean(runs)),dj_std_run=float(np.std(runs)),dj_max_run=float(np.max(runs)),
                dj_mean_pause=float(np.mean(paus)) if paus else 0.0)
def asrconf_feats(m):
    segs=m.get('segs',[]); ws=m.get('words',[])
    if not segs:
        return dict(ac_alp_mean=-2.0,ac_alp_min=-2.0,ac_alp_std=0,ac_nsp_mean=1.0,ac_nsp_max=1.0,
                    ac_cr_mean=0,ac_cr_max=0,ac_wconf_mean=0,ac_wconf_min=0,ac_wconf_std=0,ac_lowconf_rate=1.0)
    alp=[s['alp'] for s in segs]; nsp=[s['nsp'] for s in segs]; cr=[s['cr'] for s in segs]
    pr=[w[2] for w in ws] or [0.0]
    return dict(ac_alp_mean=np.mean(alp),ac_alp_min=np.min(alp),ac_alp_std=np.std(alp),
                ac_nsp_mean=np.mean(nsp),ac_nsp_max=np.max(nsp),ac_cr_mean=np.mean(cr),ac_cr_max=np.max(cr),
                ac_wconf_mean=np.mean(pr),ac_wconf_min=np.min(pr),ac_wconf_std=np.std(pr),
                ac_lowconf_rate=float(np.mean([1.0 if p<0.5 else 0.0 for p in pr])))
def build_extra(df,meta,tag):
    rows=[]
    for f in df[FILE_COL]:
        m=meta[f]; d={}
        if USE_DEJONG:   d.update(dejong_feats(m))
        if USE_ASR_CONF: d.update(asrconf_feats(m))
        rows.append(d)
    out=pd.DataFrame(rows) if rows and (USE_DEJONG or USE_ASR_CONF) else pd.DataFrame(index=range(len(df)))
    out.insert(0,FILE_COL,df[FILE_COL].values); return out
ex_tr=build_extra(train,meta_tr,'tr'); ex_te=build_extra(test,meta_te,'te')
print('dejong+asrconf dims:',ex_tr.shape[1]-1)""")

md("## 5 · Linguistic features (recomputed on turbo transcripts, cached _v4)")
code("""import nltk; [nltk.download(x,quiet=True) for x in ('punkt','averaged_perceptron_tagger','punkt_tab','averaged_perceptron_tagger_eng')]
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
    return cache_pq(f'lg_{tag}_v4.parquet',_b)
lg_tr=build_ling(tx_tr,'tr'); lg_te=build_ling(tx_te,'te')
print(lg_tr.shape)""")

md("""## 6 · GEC error features (cached _v4) — the most target-aligned signal
A grammar-error-correction model rewrites each sentence; we measure **how much it had to change**.
More edits ⇒ worse grammar. We record the overall edit rate, insertion/deletion/substitution rates,
the fraction of sentences changed, and the **worst-sentence** rate (captures a single bad clause the
clip-mean would hide).""")
code("""def build_gec(tx,tag):
    def _b():
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok=AutoTokenizer.from_pretrained(GEC_MODEL)
        mdl=AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL).to(DEVICE).eval()
        @torch.no_grad()
        def correct(sents):
            xs=[GEC_PREFIX+s for s in sents]
            inp=tok(xs,return_tensors='pt',padding=True,truncation=True,max_length=128).to(DEVICE)
            out=mdl.generate(**inp,max_length=128,num_beams=4)
            return tok.batch_decode(out,skip_special_tokens=True)
        def diff(o,c):
            a=o.split(); b=c.split(); sm=difflib.SequenceMatcher(a=a,b=b); ins=dele=sub=0
            for op,i1,i2,j1,j2 in sm.get_opcodes():
                if op=='replace': sub+=max(i2-i1,j2-j1)
                elif op=='insert': ins+=j2-j1
                elif op=='delete': dele+=i2-i1
            return ins,dele,sub,max(1,len(a))
        def feats(text):
            sents=[s.strip() for s in (nltk.sent_tokenize(text) if text else []) if s.strip()][:GEC_MAX_SENTS]
            if not sents:
                return dict(gec_edit_rate=0,gec_mean_rate=0,gec_max_rate=0,gec_frac_changed=0,
                            gec_ins_rate=0,gec_del_rate=0,gec_sub_rate=0,gec_nsents=0)
            corr=[]
            for i in range(0,len(sents),GEC_BATCH): corr+=correct(sents[i:i+GEC_BATCH])
            te=tn=ti=td=ts=0; per=[]; chg=0
            for o,c in zip(sents,corr):
                i_,d_,s_,n=diff(o,c); e=i_+d_+s_; te+=e; tn+=n; ti+=i_; td+=d_; ts+=s_
                per.append(e/n); chg+=int(e>0)
            return dict(gec_edit_rate=te/max(1,tn),gec_mean_rate=float(np.mean(per)),
                        gec_max_rate=float(np.max(per)),gec_frac_changed=chg/len(sents),
                        gec_ins_rate=ti/max(1,tn),gec_del_rate=td/max(1,tn),gec_sub_rate=ts/max(1,tn),
                        gec_nsents=len(sents))
        rows=[feats(t) for t in tqdm(tx['text'],desc=f'gec-{tag}')]
        del mdl,tok; gc.collect(); torch.cuda.empty_cache()
        out=pd.DataFrame(rows); out.insert(0,FILE_COL,tx[FILE_COL].values); return out
    return cache_pq(f'gec_{tag}_v4.parquet',_b)
if USE_GEC:
    gec_tr=build_gec(tx_tr,'tr'); gec_te=build_gec(tx_te,'te'); print('GEC dims:',gec_tr.shape[1]-1)
else:
    gec_tr=pd.DataFrame({FILE_COL:train[FILE_COL].values}); gec_te=pd.DataFrame({FILE_COL:test[FILE_COL].values})""")

md("""## 7 · Zero-shot LLM grammar judge (cached _v4) — feature + blend member
An instruction-tuned model reads the transcript and rates grammar 0–5 against the rubric. Used two
ways: as a **feature** the trees can exploit, and (SOTA pattern) as a raw **NNLS blend member** whose
weight the blend learns. It never sees the labels, so both uses are leakage-free.""")
code("""RUBRIC=("You are a strict English grammar examiner scoring spoken-English transcripts (ASR, so "
        "ignore punctuation/casing). Rate ONLY grammatical accuracy and sentence structure on a 0-5 "
        "scale: 5=high accuracy with well-controlled complex grammar; 4=good control, minor errors; "
        "3=decent but clear/frequent errors; 2=basic structures with consistent mistakes, incomplete "
        "sentences; 1=struggles with basic sentence structure. Decimals allowed. Reply with ONLY the number.")
def build_llm(tx,tag):
    def _b():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok=AutoTokenizer.from_pretrained(LLM_MODEL)
        mdl=AutoModelForCausalLM.from_pretrained(LLM_MODEL,torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32,
                                                 device_map='auto').eval()
        dbg=[0]
        @torch.no_grad()
        def judge(text):
            text=(text or '').strip()
            if not text: return np.nan
            msg=[{'role':'system','content':RUBRIC},
                 {'role':'user','content':'Transcript:\\n'+text[:2400]+'\\n\\nGrammar score (0-5):'}]
            enc=tok.apply_chat_template(msg,add_generation_prompt=True,return_tensors='pt',
                                        return_dict=True).to(mdl.device)   # dict {input_ids,attention_mask}
            out=mdl.generate(**enc,max_new_tokens=8,do_sample=False)
            dec=tok.decode(out[0][enc['input_ids'].shape[1]:],skip_special_tokens=True)
            if dbg[0]<3: print(f'  llm raw[{dbg[0]}]:',repr(dec)); dbg[0]+=1   # diagnostics: see exactly what it emits
            m=re.search(r'([0-5](?:\\.\\d+)?)',dec); return float(m.group(1)) if m else np.nan
        vals=[judge(t) for t in tqdm(tx['text'],desc=f'llm-{tag}')]
        del mdl,tok; gc.collect(); torch.cuda.empty_cache()
        out=pd.DataFrame({FILE_COL:tx[FILE_COL].values,'llm_score':vals}); return out
    if Path(f'llm_{tag}_v4.parquet').exists(): print('cache hit : llm',tag); return pd.read_parquet(f'llm_{tag}_v4.parquet')
    df=_b(); df.to_parquet(f'llm_{tag}_v4.parquet'); return df
if USE_LLM_JUDGE:
    llm_tr=build_llm(tx_tr,'tr'); llm_te=build_llm(tx_te,'te')
    both=pd.concat([llm_tr['llm_score'],llm_te['llm_score']]); cov=int(both.notna().sum())
    ymean=float(train[LABEL_COL].astype(float).mean())
    fill=float(np.nanmean(both)) if cov>0 else ymean
    if not np.isfinite(fill): fill=ymean
    llm_tr['llm_score']=llm_tr['llm_score'].fillna(fill); llm_te['llm_score']=llm_te['llm_score'].fillna(fill)
    print(f'llm judge coverage: {cov}/{len(both)}  fill={fill:.3f}')
    if cov==0:   # judge produced nothing parseable -> disable the lever instead of poisoning the blend
        print('!! LLM judge returned NO parseable scores (see raw[] above) -> disabling LLM lever this run')
        USE_LLM_JUDGE=False
        llm_tr=pd.DataFrame({FILE_COL:train[FILE_COL].values}); llm_te=pd.DataFrame({FILE_COL:test[FILE_COL].values})
else:
    llm_tr=pd.DataFrame({FILE_COL:train[FILE_COL].values}); llm_te=pd.DataFrame({FILE_COL:test[FILE_COL].values})""")

md("## 8 · SSL raw embeddings (cached — reused from v3; PCA happens in-fold in §9)")
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
if USE_SSL:
    w2_tr,w2_te=build_ssl('facebook/wav2vec2-large-960h','w2v')
    wl_tr,wl_te=build_ssl('microsoft/wavlm-large','wl')
    print('w2v',w2_tr.shape,'wavlm',wl_tr.shape)
else:
    w2_tr=w2_te=wl_tr=wl_te=None""")

md("## 9 · Model — in-fold PCA/scaling · 4 regressors (+LLM blend member) · NNLS · stability · honest calibration")
code("""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from functools import reduce
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))

# assemble aligned blocks (handcrafted = everything except raw SSL; SSL kept separate for in-fold PCA)
blocks=[ac_tr,lg_tr,ex_tr]
if USE_GEC: blocks.append(gec_tr)
if USE_LLM_JUDGE: blocks.append(llm_tr)
if USE_SSL: blocks+=[w2_tr,wl_tr]
Mtr=reduce(lambda a,b:a.merge(b,on=FILE_COL),blocks)
blocks_te=[ac_te,lg_te,ex_te]
if USE_GEC: blocks_te.append(gec_te)
if USE_LLM_JUDGE: blocks_te.append(llm_te)
if USE_SSL: blocks_te+=[w2_te,wl_te]
Mte=reduce(lambda a,b:a.merge(b,on=FILE_COL),blocks_te)
y=train.merge(Mtr[[FILE_COL]],on=FILE_COL)[LABEL_COL].astype(float).values
w2c=[c for c in Mtr.columns if c.startswith('w2v_raw')]; wlc=[c for c in Mtr.columns if c.startswith('wl_raw')]
handc=[c for c in Mtr.columns if c!=FILE_COL and c not in w2c and c not in wlc]
def _v(df,cols): return np.nan_to_num(df[cols].values.astype(np.float32))
Atr,Ate=_v(Mtr,handc),_v(Mte,handc)
Wtr,Wte=(_v(Mtr,w2c),_v(Mte,w2c)) if USE_SSL else (np.zeros((len(y),0),'float32'),np.zeros((len(Mte),0),'float32'))
Ltr,Lte=(_v(Mtr,wlc),_v(Mte,wlc)) if USE_SSL else (np.zeros((len(y),0),'float32'),np.zeros((len(Mte),0),'float32'))
bins=pd.cut(y,bins=[-0.1,1.5,2.5,3.5,4.5,5.1],labels=False)
print('handcrafted',Atr.shape[1],'| w2v raw',Wtr.shape[1],'| wavlm raw',Ltr.shape[1])

def infold_pca(Rtr,Rte,tr,dim,seed):
    if Rtr.shape[1]==0: return np.zeros((Rtr.shape[0],0)),np.zeros((Rte.shape[0],0))
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
        w2p_tr,w2p_te=infold_pca(Wtr,Wte,tr,SSL_PCA_DIM,seed)
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

# optional zero-shot LLM judge as a blend member (its OOF == its raw prediction; never trained on y)
if USE_LLM_JUDGE and LLM_AS_BLEND and 'llm_score' in Mtr.columns:
    lo=np.clip(np.nan_to_num(Mtr['llm_score'].values.astype(float),nan=float(np.mean(y))),0,5)
    lp=np.clip(np.nan_to_num(Mte['llm_score'].values.astype(float),nan=float(np.mean(y))),0,5)
    MODELS=MODELS+['llm']; oof['llm']=lo; pred['llm']=lp
# fail-safe: keep only members whose OOF+pred are fully finite (one bad member can't crash the blend)
MODELS=[m for m in MODELS if np.isfinite(oof[m]).all() and np.isfinite(pred[m]).all()]
for m in MODELS: print(f'{m:4s} OOF RMSE={rmse(y,oof[m]):.4f}  Pearson={pearsonr(y,oof[m])[0]:.4f}')

Moof=np.stack([oof[m] for m in MODELS],1); wv,_=nnls(Moof,y)
oof_blend=Moof@wv
preds_raw=np.clip(np.stack([pred[m] for m in MODELS],1)@wv,0,5)
print('NNLS weights:',{m:round(float(x),3) for m,x in zip(MODELS,wv)})
print(f'>> BLEND OOF RMSE={rmse(y,oof_blend):.4f}  Pearson={pearsonr(y,oof_blend)[0]:.4f}   (v3 was ~0.5335 / 0.9040)')

# per-fold stability (low std => generalises evenly)
fr=[]
for seed in SEEDS:
    for tr,va in StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed).split(np.zeros(len(y)),bins):
        fr.append(rmse(y[va],np.clip(oof_blend[va],0,5)))
print(f'per-fold OOF RMSE: mean={np.mean(fr):.4f}  std={np.std(fr):.4f}')

# ---- honest nested-CV isotonic calibration (monotonic: helps RMSE, preserves Pearson) ----
oob=np.clip(oof_blend,0,5); cal_oof=np.zeros_like(oob)
for tr,va in KFold(n_splits=5,shuffle=True,random_state=0).split(oob):
    iso=IsotonicRegression(out_of_bounds='clip').fit(oob[tr],y[tr]); cal_oof[va]=iso.predict(oob[va])
raw_rmse=rmse(y,oob); cal_rmse=rmse(y,cal_oof)
print(f'calibration (nested): raw OOF RMSE={raw_rmse:.4f} -> calibrated={cal_rmse:.4f}  (delta={cal_rmse-raw_rmse:+.4f})')
print(f'  Pearson raw={pearsonr(y,oob)[0]:.4f}  calibrated={pearsonr(y,cal_oof)[0]:.4f}  (monotonic -> ~unchanged)')
FINAL_ISO=IsotonicRegression(out_of_bounds='clip').fit(oob,y) if (APPLY_CALIBRATION and cal_rmse<raw_rmse) else None
preds=np.clip(FINAL_ISO.predict(preds_raw),0,5) if FINAL_ISO is not None else preds_raw
print('calibration applied to final preds:',FINAL_ISO is not None)""")

md("## 10 · Report & visualizations (interpretability is graded)")
code("""import matplotlib.pyplot as plt
oob=np.clip(oof_blend,0,5)
fig,ax=plt.subplots(2,3,figsize=(17,9.5))
# (1) predicted vs actual
ax[0,0].scatter(y,oob,s=14,alpha=0.5); ax[0,0].plot([0,5],[0,5],'r--',lw=1)
ax[0,0].set_xlabel('true'); ax[0,0].set_ylabel('OOF pred'); ax[0,0].set_title(f'Pred vs Actual  RMSE={rmse(y,oob):.3f}  r={pearsonr(y,oob)[0]:.3f}')
# (2) per-fold RMSE
ax[0,1].bar(range(len(fr)),fr); ax[0,1].axhline(np.mean(fr),color='r',ls='--')
ax[0,1].set_title(f'Per-fold OOF RMSE (mean={np.mean(fr):.3f} std={np.std(fr):.3f})'); ax[0,1].set_xlabel('fold')
# (3) score distributions
ax[0,2].hist(y,bins=20,alpha=0.6,label='true'); ax[0,2].hist(oob,bins=20,alpha=0.6,label='pred')
ax[0,2].legend(); ax[0,2].set_title('Score distribution')
# (4) residuals
res=oob-y; ax[1,0].hist(res,bins=25); ax[1,0].axvline(0,color='r',ls='--'); ax[1,0].set_title(f'Residuals (mean={res.mean():+.3f})')
# (5) binned confusion
tb=np.clip(np.round(y).astype(int),0,5); pb=np.clip(np.round(oob).astype(int),0,5)
C=np.zeros((6,6))
for a,b in zip(tb,pb): C[a,b]+=1
im=ax[1,1].imshow(C,cmap='Blues'); ax[1,1].set_xlabel('pred band'); ax[1,1].set_ylabel('true band'); ax[1,1].set_title('Binned confusion')
for a in range(6):
    for b in range(6):
        if C[a,b]: ax[1,1].text(b,a,int(C[a,b]),ha='center',va='center',fontsize=8)
# (6) illustrative feature importance (full-fit LGB over handcrafted names; interpretability only)
lgb_imp=lgb.train(lgb_params,lgb.Dataset(Atr,y),num_boost_round=400)
imp=pd.Series(lgb_imp.feature_importance(importance_type='gain'),index=handc).sort_values(ascending=False).head(18)[::-1]
ax[1,2].barh(range(len(imp)),imp.values); ax[1,2].set_yticks(range(len(imp))); ax[1,2].set_yticklabels(imp.index,fontsize=7)
ax[1,2].set_title('Top handcrafted features (LGB gain)')
plt.tight_layout(); plt.savefig('report_v4.png',dpi=110); plt.show()

print('\\n=== FEATURE-GROUP READOUT (share of LGB gain over handcrafted block) ===')
grp=lambda pre: float(imp.reindex([c for c in handc if c.startswith(pre)]).fillna(0).sum())
full=pd.Series(lgb_imp.feature_importance(importance_type='gain'),index=handc)
for name,pre in [('GEC grammar','gec_'),('LLM judge','llm_'),('De-Jong fluency','dj_'),
                 ('ASR confidence','ac_'),('linguistic ppl/err','err'),('POS','pos_')]:
    s=float(full[[c for c in handc if c.startswith(pre)]].sum())
    print(f'  {name:22s}: {100*s/full.sum():5.1f}%')

print('\\n=== WORST 10 OOF PREDICTIONS (error analysis) ===')
err=pd.DataFrame({FILE_COL:Mtr[FILE_COL].values,'true':y,'pred':np.round(oob,2),'abs_err':np.round(np.abs(oob-y),2)})
err=err.merge(tx_tr,on=FILE_COL).sort_values('abs_err',ascending=False).head(10)
for _,r in err.iterrows():
    print(f\"  {r['abs_err']:.2f} | true {r['true']:.1f} pred {r['pred']:.2f} | {r['text'][:90]}\")""")

md("## 11 · Final metrics + submission (RMSE reported as the brief requires; keyed to test.csv)")
code("""# COMPULSORY per competition rules: report training RMSE.
print(f'TRAINING (OOF, honest) RMSE = {rmse(y,np.clip(oof_blend,0,5)):.4f}')
print(f'TRAINING (OOF, honest) Pearson = {pearsonr(y,np.clip(oof_blend,0,5))[0]:.4f}')
sub=pd.DataFrame({FILE_COL:test[FILE_COL].values})
sub['label']=sub[FILE_COL].map(dict(zip(Mte[FILE_COL].values,preds))).fillna(float(np.mean(y)))
sub['label']=np.clip(sub['label'],0,5)
sub.to_csv('submission.csv',index=False)
assert len(sub)==len(test) and sub['label'].notna().all(), 'submission must have one row per test file'
print(sub.head()); print('rows:',len(sub),'range:',round(sub.label.min(),3),'..',round(sub.label.max(),3))""")

nb['cells']=cells
nbf.write(nb,'solution_v4.ipynb')
print('wrote solution_v4.ipynb with',len(cells),'cells')

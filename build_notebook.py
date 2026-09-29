"""Generates solution.ipynb for the SHL Hiring Assessment 2026 grammar-scoring competition."""
import json, nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

md("""# SHL Hiring Assessment 2026 — Grammar Scoring Engine
**Task:** Predict a continuous grammar score (0–5) for 45–60 s spoken-English audio clips.
**Metric:** Pearson correlation (primary) & RMSE.
**Data:** 769 train / 216 test `.wav` files + labels.

## Approach (why this design)
With only 769 labeled clips, fine-tuning a large audio transformer end-to-end overfits.
Instead we combine **three complementary frozen feature families** and stack a light regressor on top:

1. **Whisper ASR → linguistic features** – LanguageTool error rate, GPT-2 perplexity, TTR, length stats.
2. **Prosodic / acoustic features** – librosa MFCC & spectral stats, pitch, energy, speaking rate, pause ratio.
3. **Wav2Vec2 mean-pooled embeddings** – 1024-D self-supervised speech representations.

Modeling: 5-fold `StratifiedKFold` on binned scores → LightGBM + Ridge blend → clip to [1, 5].
Reported: **OOF Pearson + RMSE, training RMSE, residual & scatter plots.**
""")

md("## 1 · Setup")
code("""# Kaggle env already has torch, transformers, librosa. Install extras.
!pip -q install lightgbm==4.3.0 language-tool-python==2.7.1 openai-whisper==20231117 \\
                parselmouth-praat==0.4.3 soundfile==0.12.1 sentence-transformers==2.7.0 \\
                nltk==3.8.1 || true
import os, gc, json, warnings, math, re, random, subprocess
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, librosa, torch
from pathlib import Path
from tqdm.auto import tqdm
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)""")

md("## 2 · Load competition data")
code("""# On Kaggle, the input is at /kaggle/input/<comp-slug>/. When running via kagglehub locally:
# import kagglehub; DATA = Path(kagglehub.competition_download('shl-hiring-assessment-2026'))
CANDIDATES = [
    "/kaggle/input/shl-hiring-assessment-2026",
    "/kaggle/input/shl-hiring-assessment-2026/Dataset_Final",
]
DATA = next((Path(p) for p in CANDIDATES if Path(p).exists()), None)
if DATA is None:
    import kagglehub
    DATA = Path(kagglehub.competition_download('shl-hiring-assessment-2026'))
print("DATA:", DATA)
for p in sorted(DATA.rglob('*'))[:40]: print(" ", p.relative_to(DATA))""")

code("""# Auto-detect folder layout
def find(name_regex, root=DATA):
    for p in root.rglob('*'):
        if re.search(name_regex, p.name, re.I): return p
    return None
train_csv = find(r'^train\\.csv$')
test_csv  = find(r'^test\\.csv$')
subm_csv  = find(r'sample_submission')
train_wav_dir = next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_train' in p.name.lower()), None) \\
             or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('train','train_audios','audio_train')), None)
test_wav_dir  = next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_test' in p.name.lower()), None) \\
             or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('test','test_audios','audio_test')), None)
print(dict(train_csv=train_csv, test_csv=test_csv, subm=subm_csv,
           train_wav=train_wav_dir, test_wav=test_wav_dir))
train = pd.read_csv(train_csv); test = pd.read_csv(test_csv); subm = pd.read_csv(subm_csv)
LABEL_COL = 'label' if 'label' in train.columns else [c for c in train.columns if c!='filename'][0]
FILE_COL  = 'filename' if 'filename' in train.columns else train.columns[0]
print(train.shape, test.shape); train.head()""")

md("## 3 · Quick EDA")
code("""import matplotlib.pyplot as plt, seaborn as sns
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(train[LABEL_COL], bins=20, kde=True, ax=ax[0]); ax[0].set_title("Train label distribution")
train[LABEL_COL].describe().to_frame().T
""")

code("""# audio-length sanity check on a random subset
sample = train.sample(min(60, len(train)), random_state=SEED)
dur = [librosa.get_duration(path=str(train_wav_dir/f)) for f in sample[FILE_COL]]
print(f"duration s — mean {np.mean(dur):.1f}, min {np.min(dur):.1f}, max {np.max(dur):.1f}")
plt.figure(figsize=(6,3)); sns.histplot(dur, bins=20); plt.title("Audio duration (sec) — sample"); plt.show()""")

md("## 4 · Feature extraction")
md("### 4.1 Acoustic features (librosa + parselmouth)")
code("""import parselmouth
from parselmouth.praat import call

SR = 16000
def acoustic_feats(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    dur = len(y)/sr
    # librosa MFCC + delta stats
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc); d2 = librosa.feature.delta(mfcc, order=2)
    feats = {}
    for name, arr in [('mfcc',mfcc),('d1',d1),('d2',d2)]:
        feats[f'{name}_mean'] = arr.mean(); feats[f'{name}_std'] = arr.std()
        for i in range(arr.shape[0]):
            feats[f'{name}{i}_m'] = arr[i].mean(); feats[f'{name}{i}_s'] = arr[i].std()
    # spectral
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    for n,a in [('sc',sc),('sbw',sbw),('sro',sro),('zcr',zcr),('rms',rms)]:
        feats[f'{n}_m']=a.mean(); feats[f'{n}_s']=a.std()
    # silence / speaking-rate proxy
    intervals = librosa.effects.split(y, top_db=30)
    speech = sum((b-a) for a,b in intervals)/sr
    feats.update(dict(dur=dur, speech_ratio=speech/dur,
                      n_pauses=len(intervals)-1,
                      mean_pause=np.mean([intervals[i+1,0]-intervals[i,1] for i in range(len(intervals)-1)])/sr if len(intervals)>1 else 0))
    # praat pitch, jitter, shimmer
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch()
        pv = pitch.selected_array['frequency']; pv = pv[pv>0]
        feats['f0_m'] = pv.mean() if len(pv) else 0
        feats['f0_s'] = pv.std()  if len(pv) else 0
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        feats['jitter']  = call(pp, "Get jitter (local)", 0,0, 1e-4,0.02,1.3)
        feats['shimmer'] = call([snd,pp], "Get shimmer (local)", 0,0, 1e-4,0.02,1.3,1.6)
    except Exception:
        feats.update(f0_m=0, f0_s=0, jitter=0, shimmer=0)
    return feats

def extract_all(df, wav_dir, tag):
    rows = []
    for f in tqdm(df[FILE_COL], desc=f'acoustic-{tag}'):
        rows.append(acoustic_feats(str(wav_dir/f)))
    out = pd.DataFrame(rows); out.insert(0, FILE_COL, df[FILE_COL].values)
    return out

ac_tr = extract_all(train, train_wav_dir, 'train')
ac_te = extract_all(test,  test_wav_dir,  'test')
ac_tr.to_parquet('ac_tr.parquet'); ac_te.to_parquet('ac_te.parquet')
print(ac_tr.shape)""")

md("### 4.2 Whisper ASR → transcripts")
code("""import whisper
asr = whisper.load_model("medium", device=DEVICE)   # step down to 'small' if OOM
def transcribe(path):
    r = asr.transcribe(path, language='en', fp16=(DEVICE=='cuda'), condition_on_previous_text=False)
    return r['text'].strip()

def asr_df(df, wav_dir, tag):
    txts=[]
    for f in tqdm(df[FILE_COL], desc=f'asr-{tag}'):
        try: txts.append(transcribe(str(wav_dir/f)))
        except Exception as e: txts.append(''); print('asr fail',f,e)
    return pd.DataFrame({FILE_COL: df[FILE_COL].values, 'text': txts})

tx_tr = asr_df(train, train_wav_dir, 'train'); tx_tr.to_csv('tx_tr.csv', index=False)
tx_te = asr_df(test,  test_wav_dir,  'test');  tx_te.to_csv('tx_te.csv', index=False)
del asr; gc.collect(); torch.cuda.empty_cache()
tx_tr.head()""")

md("### 4.3 Linguistic features from transcripts")
code("""import nltk; [nltk.download(x, quiet=True) for x in ('punkt','averaged_perceptron_tagger','punkt_tab','averaged_perceptron_tagger_eng')]
import language_tool_python
lt = language_tool_python.LanguageTool('en-US')

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
gpt_tok = GPT2TokenizerFast.from_pretrained('gpt2')
gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE).eval()

@torch.no_grad()
def perplexity(text, stride=512, max_len=1024):
    if not text.strip(): return 200.0
    enc = gpt_tok(text, return_tensors='pt').input_ids.to(DEVICE)
    nll, n = 0.0, 0
    for i in range(0, enc.size(1), stride):
        chunk = enc[:, i:i+max_len]
        if chunk.size(1) < 2: break
        out = gpt(chunk, labels=chunk)
        nll += out.loss.item()*(chunk.size(1)-1); n += chunk.size(1)-1
    return math.exp(nll/max(n,1))

def ling_feats(text):
    text = text or ""
    words = re.findall(r"\\w+", text.lower())
    sents = nltk.sent_tokenize(text) if text else []
    n_w, n_s = max(1,len(words)), max(1,len(sents))
    errs = lt.check(text) if text else []
    pos  = nltk.pos_tag(words) if words else []
    from collections import Counter
    pc = Counter(t for _,t in pos)
    return {
        'n_words': n_w, 'n_sents': n_s,
        'avg_wlen': np.mean([len(w) for w in words]) if words else 0,
        'avg_slen': n_w/n_s,
        'ttr': len(set(words))/n_w,
        'err_rate': len(errs)/n_w,
        'n_errs': len(errs),
        'ppl': perplexity(text[:4000]),
        'log_ppl': math.log1p(perplexity(text[:4000])),
        **{f'pos_{k}': pc.get(k,0)/n_w for k in ['NN','VB','JJ','RB','PRP','DT','IN','CC','MD']},
    }

def build_ling(tx, tag):
    rows=[ling_feats(t) for t in tqdm(tx['text'], desc=f'ling-{tag}')]
    out = pd.DataFrame(rows); out.insert(0, FILE_COL, tx[FILE_COL].values); return out

lg_tr = build_ling(tx_tr, 'train'); lg_tr.to_parquet('lg_tr.parquet')
lg_te = build_ling(tx_te, 'test');  lg_te.to_parquet('lg_te.parquet')
del gpt, gpt_tok; gc.collect(); torch.cuda.empty_cache()
lg_tr.head()""")

md("### 4.4 Wav2Vec2 mean-pooled embeddings")
code("""from transformers import Wav2Vec2Processor, Wav2Vec2Model
W2V = "facebook/wav2vec2-large-960h"
proc = Wav2Vec2Processor.from_pretrained(W2V)
w2v  = Wav2Vec2Model.from_pretrained(W2V).to(DEVICE).eval()

@torch.no_grad()
def w2v_embed(path, chunk_s=20):
    y,_ = librosa.load(path, sr=16000, mono=True)
    chunks = [y[i:i+chunk_s*16000] for i in range(0, len(y), chunk_s*16000)] or [y]
    embs=[]
    for c in chunks:
        inp = proc(c, sampling_rate=16000, return_tensors='pt').input_values.to(DEVICE)
        h = w2v(inp).last_hidden_state.mean(1).squeeze(0).cpu().numpy()
        embs.append(h)
    return np.mean(embs, 0)

def emb_df(df, wav_dir, tag):
    E = np.stack([w2v_embed(str(wav_dir/f)) for f in tqdm(df[FILE_COL], desc=f'w2v-{tag}')])
    out = pd.DataFrame(E, columns=[f'w2v_{i}' for i in range(E.shape[1])])
    out.insert(0, FILE_COL, df[FILE_COL].values); return out

w2_tr = emb_df(train, train_wav_dir, 'train'); w2_tr.to_parquet('w2_tr.parquet')
w2_te = emb_df(test,  test_wav_dir,  'test');  w2_te.to_parquet('w2_te.parquet')
del w2v, proc; gc.collect(); torch.cuda.empty_cache()
print(w2_tr.shape)""")

md("## 5 · Assemble feature matrices")
code("""def merge(a,b,c): return a.merge(b,on=FILE_COL).merge(c,on=FILE_COL)
X_tr_full = merge(ac_tr, lg_tr, w2_tr)
X_te_full = merge(ac_te, lg_te, w2_te)
y = train.merge(X_tr_full[[FILE_COL]], on=FILE_COL)[LABEL_COL].astype(float).values
X_tr = X_tr_full.drop(columns=[FILE_COL]).astype(np.float32).fillna(0).values
X_te = X_te_full.drop(columns=[FILE_COL]).astype(np.float32).fillna(0).values
print(X_tr.shape, X_te.shape)""")

md("## 6 · Cross-validated LightGBM + Ridge blend")
code("""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

bins = pd.cut(y, bins=[-0.1,1.5,2.5,3.5,4.5,5.1], labels=False)
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(y)); oof_rdg = np.zeros(len(y))
pred_lgb = np.zeros(len(X_te)); pred_rdg = np.zeros(len(X_te))
lgb_params = dict(objective='regression', metric='rmse',
                  learning_rate=0.03, num_leaves=31, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=8,
                  lambda_l2=1.0, verbose=-1, seed=SEED)

for fold,(tr,va) in enumerate(skf.split(X_tr, bins)):
    dtr = lgb.Dataset(X_tr[tr], y[tr]); dva = lgb.Dataset(X_tr[va], y[va])
    m = lgb.train(lgb_params, dtr, num_boost_round=4000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof_lgb[va] = m.predict(X_tr[va]); pred_lgb += m.predict(X_te)/skf.n_splits

    sc = StandardScaler().fit(X_tr[tr])
    r = Ridge(alpha=8.0).fit(sc.transform(X_tr[tr]), y[tr])
    oof_rdg[va] = r.predict(sc.transform(X_tr[va]))
    pred_rdg  += r.predict(sc.transform(X_te))/skf.n_splits
    print(f'fold{fold}  lgb rmse={np.sqrt(mean_squared_error(y[va],oof_lgb[va])):.3f}'
          f'  ridge rmse={np.sqrt(mean_squared_error(y[va],oof_rdg[va])):.3f}')

# blend weight tuned on OOF
best_w, best_r = 0.5, -1
for w in np.linspace(0,1,21):
    p = w*oof_lgb + (1-w)*oof_rdg
    r = pearsonr(y, p)[0]
    if r>best_r: best_r, best_w = r, w
print(f'best blend w={best_w:.2f}  OOF pearson={best_r:.4f}')

oof   = best_w*oof_lgb   + (1-best_w)*oof_rdg
preds = best_w*pred_lgb  + (1-best_w)*pred_rdg
preds = np.clip(preds, 1, 5)

oof_rmse = np.sqrt(mean_squared_error(y, oof))
oof_pear = pearsonr(y, oof)[0]
print(f'>> OOF RMSE={oof_rmse:.4f}   Pearson={oof_pear:.4f}')""")

md("## 7 · Training-set RMSE (required by rules)")
code("""# Refit on full data with LGB best_iter approx = 1500 for a training-fit metric
full_m = lgb.train(lgb_params, lgb.Dataset(X_tr, y), num_boost_round=1500)
train_pred = np.clip(best_w*full_m.predict(X_tr) + (1-best_w)*Ridge(alpha=8.0)
                     .fit(StandardScaler().fit_transform(X_tr), y)
                     .predict(StandardScaler().fit_transform(X_tr)), 1, 5)
train_rmse = np.sqrt(mean_squared_error(y, train_pred))
train_pear = pearsonr(y, train_pred)[0]
print(f'TRAIN RMSE = {train_rmse:.4f}   |   TRAIN Pearson = {train_pear:.4f}')""")

md("## 8 · Diagnostic plots")
code("""fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].scatter(y, oof, alpha=.6); ax[0].plot([1,5],[1,5],'r--')
ax[0].set(xlabel='true', ylabel='oof pred', title=f'OOF scatter  r={oof_pear:.3f}')
sns.histplot(oof - y, bins=25, kde=True, ax=ax[1]); ax[1].set_title('OOF residuals')
plt.tight_layout(); plt.show()

imp = pd.Series(full_m.feature_importance('gain'),
                index=X_tr_full.drop(columns=[FILE_COL]).columns).sort_values(ascending=False)
imp.head(30).plot(kind='barh', figsize=(6,8)); plt.gca().invert_yaxis()
plt.title('Top-30 LightGBM feature importance (gain)'); plt.show()""")

md("## 9 · Build submission")
code("""out = X_te_full[[FILE_COL]].copy(); out[LABEL_COL] = preds
# align to sample_submission order/columns
sub = subm.copy()
target_col = [c for c in sub.columns if c != FILE_COL][0]
sub = sub[[FILE_COL]].merge(out.rename(columns={LABEL_COL: target_col}), on=FILE_COL, how='left')
sub[target_col] = sub[target_col].fillna(y.mean())
sub.to_csv('submission.csv', index=False)
print(sub.head()); print('rows:', len(sub))""")

md("""## 10 · Report

**Pipeline**
1. Whisper-medium ASR → 216+769 English transcripts.
2. Three feature blocks — acoustic (~130-D), linguistic (~20-D), Wav2Vec2 embeddings (1024-D).
3. 5-fold StratifiedKFold on binned grammar score.
4. LightGBM regressor + Ridge over standardized features; blend weight chosen by OOF Pearson.
5. Predictions clipped to `[1, 5]`.

**Metrics**
- OOF RMSE / Pearson printed in §6.
- Full-fit training RMSE printed in §7 (competition requirement).

**Why it works on 769 samples**
- Frozen pretrained representations avoid over-fitting.
- Grammar-specific linguistic signals (LanguageTool error rate, GPT-2 perplexity) directly encode the target rubric.
- Stratified CV + blend + clipping gives a stable, well-calibrated regressor.

**How to push further if time permits**
- Add HuBERT-large or WavLM-large embeddings as another block and blend.
- Fine-tune a small regression head on Wav2Vec2 with strong dropout (only if OOF improves).
- Pseudo-label the test set and iterate one round.
""")

nb['cells'] = cells
nbf.write(nb, 'solution.ipynb')
print('wrote solution.ipynb with', len(cells), 'cells')

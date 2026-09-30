"""Generates finetune.ipynb — end-to-end WavLM fine-tune for SHL 2026 grammar scoring.

Separate from solution.ipynb (your safe 0.44). Regularised hard for 769 rows:
  WavLM-base-plus, frozen CNN feature-encoder, learnable per-layer weights + mean-pool,
  small dropout head, 16s random-crop augmentation, layer-wise LR, SmoothL1 loss,
  5-fold CV with best-epoch selection, TTA at inference. Outputs its own OOF RMSE,
  submission_finetune.csv, and oof/pred arrays so we can blend it with the frozen stack.

Upload to Kaggle (GPU + Internet on) -> Run All. Judge by the printed OOF RMSE vs your
frozen 0.556; then blend (see last cell).
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# SHL 2026 — WavLM end-to-end fine-tune
The ceiling play: train the speech model itself, not just frozen features. Kept separate from
your 0.44 `solution.ipynb`. **Judge by the OOF RMSE it prints** (compare to your frozen 0.556).
Even if it only ties, it's a *diverse* model — blending it into your stack is where the gain is.
Regularisation for 769 rows: small backbone, frozen CNN, learnable layer weights, 16 s random
crops, layer-wise LR, best-epoch early stop, TTA inference.
""")

md("## 1 · Setup + config")
code("""import os, gc, math, random, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, librosa, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

CFG = dict(
    MODEL   = 'microsoft/wavlm-base-plus',   # small on purpose (94M) — large overfits 769 rows
    CROP_S  = 16,      # seconds per training crop (clips are 45-60s -> strong augmentation)
    EPOCHS  = 8,
    BATCH   = 8,
    LR_BODY = 1e-5,    # backbone (low)
    LR_HEAD = 1e-3,    # layer-weights + head (high)
    WD      = 1e-4,
    FREEZE_LAYERS = 6, # freeze bottom N transformer layers too (anti-overfit; base-plus has 12)
    FOLDS   = 5,
    SEEDS   = [42],    # first run: 1 seed (~1h). Once it's clean, use [42,1337,2025] for a stable OOF.
)
SEED = CFG['SEEDS'][0]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CROP = CFG['CROP_S']*16000
print('device:', DEVICE, '| config:', CFG)""")

md("## 2 · Load data")
code("""import re
CANDIDATES = ["/kaggle/input/shl-hiring-assessment-2026",
              "/kaggle/input/shl-hiring-assessment-2026/Dataset_Final"]
DATA = next((Path(p) for p in CANDIDATES if Path(p).exists()), None)
if DATA is None:
    import kagglehub; DATA = Path(kagglehub.competition_download('shl-hiring-assessment-2026'))
def find(rgx):
    for p in DATA.rglob('*'):
        if re.search(rgx, p.name, re.I): return p
train_csv=find(r'^train\\.csv$'); test_csv=find(r'^test\\.csv$')
train_wav_dir = next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_train' in p.name.lower()), None) \\
             or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('train','train_audios','audio_train')), None)
test_wav_dir  = next((p for p in DATA.rglob('*') if p.is_dir() and 'audios_test' in p.name.lower()), None) \\
             or next((p for p in DATA.rglob('*') if p.is_dir() and p.name.lower() in ('test','test_audios','audio_test')), None)
train = pd.read_csv(train_csv); test = pd.read_csv(test_csv)
LABEL_COL = 'label' if 'label' in train.columns else [c for c in train.columns if c!='filename'][0]
FILE_COL  = 'filename' if 'filename' in train.columns else train.columns[0]
y = train[LABEL_COL].astype(float).values
bins = pd.cut(y, bins=[-0.1,1.5,2.5,3.5,4.5,5.1], labels=False)
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
print(train.shape, test.shape, '| wav dirs:', train_wav_dir, test_wav_dir)""")

md("## 3 · Model, dataset, inference helpers")
code("""from transformers import AutoFeatureExtractor, AutoModel
fe = AutoFeatureExtractor.from_pretrained(CFG['MODEL'])

def _feat(wav):                                  # -> 1-D float tensor of length CROP
    return fe(wav, sampling_rate=16000, return_tensors='pt').input_values[0]

class AudioDS(torch.utils.data.Dataset):
    def __init__(self, files, wav_dir, ys=None, train=False):
        self.files=list(files); self.dir=wav_dir; self.ys=ys; self.train=train
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        y_,_ = librosa.load(str(self.dir/self.files[i]), sr=16000, mono=True)
        if len(y_) >= CROP:
            st = np.random.randint(0, len(y_)-CROP+1) if self.train else (len(y_)-CROP)//2
            y_ = y_[st:st+CROP]
        else:
            y_ = np.pad(y_, (0, CROP-len(y_)))
        t = torch.tensor(self.ys[i] if self.ys is not None else 0.0, dtype=torch.float32)
        return _feat(y_), t

def collate(b):
    xs,ts = zip(*b); return torch.stack(xs), torch.stack(ts)   # fixed length -> plain stack

class WavLMRegressor(nn.Module):
    def __init__(self, name, dropout=0.1):
        super().__init__()
        self.bb = AutoModel.from_pretrained(name, output_hidden_states=True)
        self.bb.freeze_feature_encoder()                          # freeze CNN
        for p in self.bb.encoder.layers[:CFG['FREEZE_LAYERS']].parameters():
            p.requires_grad = False                               # freeze bottom transformer layers
        L = self.bb.config.num_hidden_layers + 1; H = self.bb.config.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(L))               # learnable layer weights
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(H,256), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(256,1))
    def forward(self, x):
        hs = torch.stack(self.bb(x).hidden_states, 0)             # [L,B,T,H]
        h  = (hs * torch.softmax(self.layer_w,0).view(-1,1,1,1)).sum(0).mean(1)  # weighted layers -> mean time
        return self.head(h).squeeze(-1)

@torch.no_grad()
def predict_tta(net, files, wav_dir, max_win=4):
    '''Average predictions over non-overlapping 16 s windows -> uses the whole clip.'''
    net.eval(); out=[]
    for f in files:
        y_,_ = librosa.load(str(wav_dir/f), sr=16000, mono=True)
        if len(y_) < CROP: y_ = np.pad(y_, (0, CROP-len(y_)))
        starts = list(range(0, len(y_)-CROP+1, CROP))[:max_win] or [0]
        xb = torch.stack([_feat(y_[s:s+CROP]) for s in starts]).to(DEVICE)
        with torch.cuda.amp.autocast():
            p = net(xb).float().cpu().numpy()
        out.append(float(np.mean(p)))
    return np.array(out)

print('model + helpers ready')""")

md("## 4 · Cross-validated training")
code("""from sklearn.model_selection import StratifiedKFold

def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_dl = torch.utils.data.DataLoader(AudioDS(train[FILE_COL].values[tr_idx], train_wav_dir, y[tr_idx], True),
                                        batch_size=CFG['BATCH'], shuffle=True, collate_fn=collate, num_workers=2, drop_last=False)
    va_dl = torch.utils.data.DataLoader(AudioDS(train[FILE_COL].values[va_idx], train_wav_dir, y[va_idx], False),
                                        batch_size=CFG['BATCH'], collate_fn=collate, num_workers=2)
    net = WavLMRegressor(CFG['MODEL']).to(DEVICE)
    head = list(net.head.parameters()) + [net.layer_w]
    body = [p for p in net.bb.parameters() if p.requires_grad]   # only unfrozen backbone params
    opt  = torch.optim.AdamW([{'params':body,'lr':CFG['LR_BODY']},
                              {'params':head,'lr':CFG['LR_HEAD']}], weight_decay=CFG['WD'])
    steps = max(1, len(tr_dl))*CFG['EPOCHS']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG['LR_BODY'],CFG['LR_HEAD']],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    best_rmse, best_state = 1e9, None
    for ep in range(CFG['EPOCHS']):
        net.train()
        for x,t in tr_dl:
            x,t = x.to(DEVICE), t.to(DEVICE); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = F.smooth_l1_loss(net(x), t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        # quick val (center crop) for epoch selection
        net.eval(); vp=[]
        with torch.no_grad():
            for x,_ in va_dl:
                with torch.cuda.amp.autocast(): vp.append(net(x.to(DEVICE)).float().cpu().numpy())
        r = rmse(y[va_idx], np.clip(np.concatenate(vp),0,5))
        if r < best_rmse:
            best_rmse = r; best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        print(f'    ep{ep} val_rmse(center)={r:.4f}')
    net.load_state_dict(best_state)
    oof_va  = predict_tta(net, train[FILE_COL].values[va_idx], train_wav_dir)   # TTA on full clips
    pred_te = predict_tta(net, test[FILE_COL].values, test_wav_dir)
    del net; gc.collect(); torch.cuda.empty_cache()
    return oof_va, pred_te

oof_e2e  = np.zeros(len(y))
pred_e2e = np.zeros(len(test))
fold_rmses = []
for seed in CFG['SEEDS']:
    skf = StratifiedKFold(n_splits=CFG['FOLDS'], shuffle=True, random_state=seed)
    for k,(tr_idx,va_idx) in enumerate(skf.split(np.arange(len(y)), bins)):
        print(f'seed {seed} fold {k}')
        o,p = train_fold(tr_idx, va_idx, seed)
        oof_e2e[va_idx] += o/len(CFG['SEEDS'])
        pred_e2e        += p/(CFG['FOLDS']*len(CFG['SEEDS']))
        fr = rmse(y[va_idx], np.clip(o,0,5)); fold_rmses.append(fr)
        print(f'   -> fold TTA rmse = {fr:.4f}')
oof_e2e_c = np.clip(oof_e2e,0,5)
print(f'\\n>> FINE-TUNE OOF RMSE = {rmse(y, oof_e2e_c):.4f}   (your frozen stack = 0.556)')
print(f'   per-fold: mean={np.mean(fold_rmses):.4f}  std={np.std(fold_rmses):.4f}  '
      f'(low std = generalises evenly; high std = unstable/overfit-prone)')""")

md("## 5 · Save submission + arrays")
code("""sub = pd.DataFrame({FILE_COL: test[FILE_COL].values})
sub['label'] = np.clip(pred_e2e, 0, 5)
sub.to_csv('submission_finetune.csv', index=False)
np.save('oof_e2e.npy', oof_e2e_c); np.save('pred_e2e.npy', np.clip(pred_e2e,0,5))
print(sub.head()); print('rows:', len(sub), ' range:', round(sub.label.min(),3), '..', round(sub.label.max(),3))
print('saved: submission_finetune.csv, oof_e2e.npy, pred_e2e.npy')""")

md("""## 6 · Next: blend with the frozen 0.44 stack (do NOT submit the fine-tune alone)
A fine-tune on 769 rows is overfit-prone standalone — its value is as a **diverse, small-weight
blend member**. Watch the per-fold **std** above: if it's high, the model is unstable and we
lower its blend weight or discard it. The gain generalises only through the blend. To do it:
1. In `solution.ipynb`, after §6, add two lines to save its arrays:
   `np.save('oof_frozen.npy', oof_blend); np.save('pred_frozen.npy', preds)`
2. Then blend on OOF RMSE (search one weight):
```python
of=np.load('oof_frozen.npy'); pf=np.load('pred_frozen.npy')
oe=np.load('oof_e2e.npy');    pe=np.load('pred_e2e.npy')
best=None
for a in np.linspace(0,1,41):
    e=rmse(y, a*of+(1-a)*oe)
    if best is None or e<best[0]: best=(e,a)
a=best[1]; print(f'blend a(frozen)={a:.2f}  OOF RMSE={best[0]:.4f}')
final=np.clip(a*pf+(1-a)*pe,0,5)
pd.DataFrame({FILE_COL:test[FILE_COL].values,'label':final}).to_csv('submission_blend.csv',index=False)
```
Submit `submission_blend.csv` only if its OOF beats both models. Keep the 0.44 as your safe final.
""")

nb['cells']=cells
nbf.write(nb,'finetune.ipynb')
print('wrote finetune.ipynb with', len(cells), 'cells')

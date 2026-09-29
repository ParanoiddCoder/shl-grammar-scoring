# SHL Hiring Assessment 2026 — Grammar Scoring Engine

Predict a continuous **grammar score (0–5)** for 45–60 s spoken-English `.wav` clips.
[Kaggle competition](https://www.kaggle.com/competitions/shl-hiring-assessment-2026) · metric: **RMSE** (Pearson secondary).

- **Data:** 769 train / 216 test clips.
- **Best public LB:** 0.5275 RMSE (raw blend).
- **OOF:** Pearson 0.858, RMSE 0.639.

## Approach

With only 769 labeled clips, fine-tuning a large audio transformer end-to-end overfits.
Instead we stack a light regressor on top of **three complementary frozen feature families**:

1. **Acoustic** (librosa) — 20 MFCC + Δ + ΔΔ mean/std, spectral centroid/bandwidth/rolloff, ZCR, RMS, duration, speech ratio, pause stats. ~130-D.
2. **ASR → linguistic** — `openai/whisper-small` transcripts → GPT-2 perplexity, LanguageTool error rate, TTR, sentence/word length, POS ratios. ~20-D.
3. **Wav2Vec2** — `facebook/wav2vec2-large-960h` mean-pooled embeddings. 1024-D.

**Model:** 5-fold `StratifiedKFold` on binned score → LightGBM + Ridge blend (≈0.8/0.2 by OOF) → clip to `[0, 5]`.

## Repo layout

| Path | What |
|---|---|
| `solution.ipynb` | End-to-end Kaggle notebook (GPU T4). |
| `build_notebook.py` | Regenerates the notebook. |
| `submissions/` | Best scored submissions. |

## Notes / gotchas

- Submission must be keyed against `test.csv`'s 216 filenames — **not** `sample_submission.csv` (stale, 204 mismatched rows).
- Predictions clipped to `[0, 5]` (train has ~37 label-0 silent/rejected clips).
- On Kaggle Py3.12, `openai-whisper` and `praat-parselmouth` wheels fail to build — use HF Transformers Whisper; praat is optional and gated.
- Wav2Vec2: filter chunks `< 16000` samples and wrap in try/except to avoid a conv-kernel crash.

## Run

Runs top-to-bottom on a Kaggle GPU (T4) notebook with the competition dataset attached.

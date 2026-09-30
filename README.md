# SHL Hiring Assessment 2026 — Grammar Scoring

Predict a continuous grammar score (1–5 MOS) for 45–60 second spoken-English `.wav` clips.
[Kaggle competition](https://www.kaggle.com/competitions/shl-hiring-assessment-2026) · 769 train / 216 test · scored on RMSE and Pearson correlation.

The whole pipeline runs top-to-bottom on a Kaggle GPU (T4) notebook. The final notebook is
[`solution_v6.ipynb`](solution_v6.ipynb); everything before it is the version history that got me there.

## Results

The submission I'm using is `B pos-732` (trained on the 732 in-range labels, predictions clipped to [1, 5]):

| Metric | Value |
|---|---|
| Held-out RMSE (random CV) | 0.534 |
| Held-out RMSE (duration-matched to test) | 0.580 |
| Held-out Pearson | 0.851 |
| Best public LB (v5) | 0.4263 |

A note on the numbers: the held-out figures are measured with the blend weights **and** the calibration
fit inside each fold (nothing scored on rows it was trained on), so they read a little worse than the
in-sample numbers earlier versions reported — same model, honest ruler. Because v5 and v6 measure their
CV differently, the only fair cross-version comparison is the leaderboard.

![Held-out predictions, residuals, score distribution, confusion, and feature importances](report_v6.png)

## Approach

With only ~730 usable labels, fine-tuning a large audio model end-to-end just overfits. So I freeze
several feature extractors and stack a light regressor on top of them. The feature families:

- **ASR confidence** (from Whisper decoding) — by far the strongest single signal.
- **Verbatim CTC transcript** (`facebook/wav2vec2-large-960h`, greedy) — Whisper "cleans up" grammar,
  so a raw CTC transcript recovers the error signal Whisper smooths away.
- **Whisper transcript → linguistics** — perplexity, error rate, type-token ratio, sentence/word length, POS ratios.
- **Fluency** (De-Jong style) — pause and speech-rate features.
- **Acoustic** — MFCCs, spectral stats, duration.
- **SSL embeddings** — wav2vec2 + WavLM, PCA-compressed.
- **GEC and an optional LLM judge** — kept for completeness; they turned out to add very little (see below).

**Model:** a stack of LightGBM (L2 and Huber), Ridge, SVR (RBF) and ExtraTrees, combined with a
non-negative least-squares blend and an isotonic calibration on top. Predictions clipped to [1, 5].

## What's different in v6

v6 keeps the exact feature pipeline from v5 (so every cached feature is reused) and rebuilds the
evaluation and target handling around two things I verified in the data:

1. **37 training labels are 0** — a contiguous, louder, batch-normalised block outside the 1–5 rubric.
   The test IDs and audio look nothing like it, so I treat those 37 as an anomaly. v6 trains with and
   without them and scores both on the same in-range targets, instead of assuming.
2. **Train is 60s-dominant, test is 45s-dominant**, and duration correlates with score. A plain random
   CV isn't test-like, so v6 also reports an RMSE reweighted to match the test's duration mix.

On top of that, the blend weights and calibration are fit inside each fold (held out), and the config
choice (all-labels vs in-range, grammar specialist, CTC ablation) is compared honestly rather than
picked by the lowest number.

## What the data said

- Every config landed within ~0.002 RMSE of the others (fold noise is ~0.011), so dropping the zeros,
  adding a grammar specialist, or removing CTC didn't meaningfully move the score. The model is stable
  and near its ceiling for this feature set.
- ASR confidence carries ~29% of the tree model's gain; the verbatim-CTC block ~9%. The expensive
  extras — the LLM judge (0.5%), GEC (1.3%), Whisper↔CTC divergence (1.6%) — contribute almost nothing.
- The honest blend leans on the two linear models (SVR + Ridge ≈ 0.79 of the weight); the tree models
  add little on top.
- Where it struggles: it compresses the range (rarely predicts a clean 1 or 5), it's weaker on the
  short clips that dominate the test set, and it over-rates fluent-sounding but ungrammatical answers.

## Repo layout

| Path | What |
|---|---|
| `solution_v6.ipynb` | Final notebook — honest evaluation + data fixes. Run this. |
| `build_v6.py` | Regenerates `solution_v6.ipynb` (splices onto the v5 pipeline). |
| `solution_v2..v5.ipynb`, `build_v2..v5.py` | Version history (feature work leading up to v6). |
| `solution.ipynb`, `build_notebook.py` | The original v1 baseline. |
| `finetune.ipynb`, `build_finetune.py` | End-to-end WavLM experiment (overfits 769 rows — kept as a record). |
| `EXPERIMENTS.md` | Running log of what I tried and what it scored. |
| `submission.csv` | Final submission (216 rows, keyed to `test.csv`). |
| `report_v6.png` | The visualisations from the notebook's report section. |

## How to run

1. New Kaggle notebook → upload `solution_v6.ipynb`.
2. Add the `shl-hiring-assessment-2026` competition dataset, turn on the GPU T4 accelerator and Internet.
3. (Optional, for a fast rerun) also add a saved version of the v5 notebook's output so the cached
   features are reused instead of recomputed.
4. Run All. It writes `submission.csv` (216 rows) and the report figure.

## Notes / gotchas

- Key the submission against `test.csv`'s 216 filenames, **not** `sample_submission.csv` (it has 204 mismatched rows).
- Predictions are clipped to [1, 5]; the test set has no zero labels.
- On Kaggle Py3.12, the `openai-whisper` and `praat-parselmouth` wheels fail to build — use faster-whisper / HF Transformers instead.
- For wav2vec2/CTC, filter out chunks shorter than the conv kernel and wrap inference in try/except.

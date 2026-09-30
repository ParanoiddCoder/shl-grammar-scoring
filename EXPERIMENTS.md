# SHL 2026 — experiment ladder

**Process (unchanged):** decide on **OOF RMSE**, not public LB (~108 rows; moves <0.01 are
noise). Change **one CONFIG flag per run**. Paste back §6 per-model + blend OOF RMSE, §6b
nested-calibration deltas, §8 plots, and the LB. No number is trusted until a run backs it.

All knobs live in the **CONFIG** cell (§0). Defaults reproduce the frozen-stack baseline so
run #1 validates the reconstruction before anything new is switched on.

> Numbers below are **hypotheses** (direction + rough magnitude from priors), not promises.
> Keep a flag only if OOF RMSE improves. If a flag is neutral/worse, revert it and move on.

| # | Change (from previous kept-best) | Risk | Hypothesis on OOF RMSE | Cost |
|---|----------------------------------|------|------------------------|------|
| 1 | **defaults** — validate reconstruction | — | lands ~0.55–0.56 (anchor) | full run |
| 2 | `SSL_LAYER_AGG='mean_all'` | low | −0.005 … −0.02 | re-extract SSL |
| 3 | `+ SSL_POOL='meanstd'` | low | −0.003 … −0.015 | re-extract SSL |
| 4 | `+ SSL_PCA_DIM=64` | low | −0.005 … +0.005 (marginal) | §5 onward |
| 5 | `+ USE_DISFLUENCY=True` | low | −0.003 … −0.015 | §4.3 onward |
| 6 | `+ USE_GEC=True` | med | −0.01 … −0.03 (ASR-noise confound) | +GEC pass |
| 7 | `ASR_MODEL='openai/whisper-medium'` | med | −0.01 … −0.03 (better transcripts help 5,6) | re-run ASR + text |
| 8 | `APPLY_CALIBRATION=True` **iff** §6b nested Δ clearly negative | med | −0.005 … −0.015 | §6b onward |
| 9 | `RUN_E2E=True` (own ~1–2 h run) | high | −0.03 … −0.08 **or** worse (overfit) | long |

## Why these, in this order
- **2–4 (SSL):** last-hidden-layer mean-pool throws away most of the encoder. Averaging all
  layers + adding time-**std** captures fluency variance and mid-layer content that the last
  (ASR-CTC) layer suppresses. Cheapest real signal, lowest risk → do first.
- **5–6 (grammar):** the target is grammar, yet top features are acoustic/SSL. Disfluency
  (repeats, fillers, subordination, MATTR) and a **GEC edit-rate** are the most target-aligned
  signals in the whole pipeline and are nearly free. GEC's confound: ASR errors get "corrected"
  too — that's why it pairs with **7**.
- **7 (ASR):** every text feature (ppl, LanguageTool, disfluency, GEC) rides on transcript
  quality. whisper-medium lowers WER; expect it to lift 5 and 6 together. Re-extract text after.
- **8 (calibration):** the OOF scatter shrinks at the extremes. Some of that is Bayes-optimal
  for RMSE and *cannot* be recovered; §6b measures the recoverable part with a leakage-free
  nested CV. Apply only if the nested delta is clearly negative — never chase the visible plot.
- **9 (end-to-end):** the genuine ceiling below ~0.42. On 769 rows it is the highest-variance
  move; the cell is regularised hard (WavLM-base-plus, frozen CNN, layer-weights + mean-pool +
  dropout head, Huber, layer-wise LR, 15 s crop aug, best-epoch). Its OOF is blended with the
  frozen stack, so it can only help the ensemble, not replace it. **This is the one cell not
  verifiable offline — expect to iterate on it.**

## v4 MEASURED RESULT (run 2026-09-30)

| pipeline | OOF RMSE | Pearson | public LB |
|----------|----------|---------|-----------|
| v3 (prior best) | 0.5335 | 0.9040 | 0.4464 |
| v4 blend (raw)  | 0.5420 | 0.9006 | — |
| **v4 blend (calibrated)** | **0.5281** | **0.9045** | **0.4354** |

Both OOF and public LB improved in the same direction -> v4 is genuinely (if modestly) the best.
per-fold OOF std = 0.0315 (stable). **Finals locked: v4 (0.4354) + the 0.44/v3 safety net.**

**What actually drove the gain (feature-group share of LGB gain over handcrafted block):**
`ASR confidence 23.4%` (top feats ac_wconf_mean/std — Whisper word-confidence tracks fluency),
`De-Jong fluency 2.1%`, `POS 1.6%`, `GEC grammar 0.9%`, `linguistic 0.3%`, **`LLM judge 0.0%`**.
Calibration (nested isotonic) was the single biggest lever: -0.0124 RMSE, Pearson unchanged/up.

**Key negative findings (don't repeat blindly):**
- The grammar-error levers I predicted (GEC edit-rate, LLM judge) contributed ~nothing. Root cause:
  **Whisper normalises grammar in its transcripts** — it outputs fluent well-formed text even from
  ungrammatical speech, washing out the error signal before GEC/LLM see it. Confirmed empirically.
- Qwen2.5-3B is a poor zero-shot grammar judge here (OOF 1.266 / Pearson 0.40; NNLS weight 0).
  A stronger judge (7B+) or a disfluency-preserving ASR would be the only way to revive this lever.
- The real, transferable win is **ASR word-confidence + De-Jong fluency + honest calibration**.

## v4 ladder — grammar-signal pipeline (`build_v4.py` -> `solution_v4.ipynb`)

Motivation (literature-backed): the target is **grammar**, yet v3 spends ~95% of its feature
budget on *how speech sounds* (MFCC/spectral + 4096 raw SSL dims) and almost nothing on *whether
it is grammatical* -> that is the OOF ~0.53 ceiling. Sources:
- "Back to grammar" (Speech Communication 2023): spoken GEC -> grammatical features -> proficiency.
- Essay-scoring MTL (arXiv 2406.08817): GECToR error frequencies as scorer inputs.
- L2-speaking SOTA (arXiv 2608.26137): interpretable fluency composite + zero-shot LLM judge -> p=0.818.
- Shortcut/fairness (arXiv 2607.16085): raw SSL dims encode accent shortcuts -> private-set risk.

v4 keeps v3's leakage-free backbone and adds, each behind a CONFIG flag, each cached, each judged
on in-fold OOF RMSE + per-fold std:

| # | Lever (CONFIG flag) | Evidence | Hypothesis on OOF RMSE | Risk |
|---|---------------------|----------|------------------------|------|
| 1 | `ASR_MODEL` = faster-whisper large-v3-turbo | rides under 2-5 | multiplies every text feature | none |
| 2 | `USE_DEJONG` fluency (run length, pause ratio, long-pause rate, speech/artic rate) | p=0.764 composite | -0.005 … -0.02 | low |
| 3 | `USE_ASR_CONF` (avg_logprob / no_speech / compression / word prob) | De-Jong composite | small, ~free | none |
| 4 | `USE_GEC` edit-rate + ins/del/sub + worst-sentence rate | Back-to-grammar; essay MTL | **-0.01 … -0.04 (biggest lever)** | low |
| 5 | `USE_LLM_JUDGE` zero-shot 0-5 (feature + NNLS member) | p=0.818 SOTA pattern | -0.01 … -0.03, lifts Pearson | low |
| 6 | `APPLY_CALIBRATION` nested isotonic (monotonic) | two-metric LB | -0.005 … -0.015 RMSE, Pearson ~flat | low |
| + | §10 report/viz (scatter, per-fold, confusion, feature-group readout, error analysis) | brief grades Interpretability | deliverable score, not RMSE | none |

Caches: ASR-independent blocks (`ac_*`, `w2v_*`, `wl_*`) are **reused from v3 runs**; every
transcript-dependent artifact is written with a `_v4` suffix (`asrmeta_*_v4.json`, `lg_*_v4`,
`gec_*_v4`, `llm_*_v4`) so v3 and v4 never collide. Turn one flag at a time; keep only OOF wins.

## Final-submission strategy (Kaggle picks 2)
1. The **best-OOF** config from this ladder.
2. A **known-good 0.44** `submission.csv` held in reserve (in `~/Downloads`, the 216-row run —
   `submission (6).csv` is the newest correct one; confirm which scored 0.44 on the LB and
   archive it under `submissions/`).

## Reproducibility note (important)
The 0.44 pipeline was **never committed** — the repo's old `solution.ipynb` still keyed to
`sample_submission` (that's why some runs wrote only **204** rows instead of 216), tuned the
blend on Pearson, and clipped [1,5]. This rebuilt `solution.ipynb` (generated by
`build_notebook.py`) is now the source of truth: upload it to Kaggle, Run All, submit
`submission.csv`. `git` again reflects what actually runs.

## v5 ladder — verbatim-transcript grammar recovery (`build_v5.py` -> `solution_v5.ipynb`)

**Diagnosis from v4's OWN readout:** the pipeline measures how speech *sounds* (ASR confidence
23.4%, De-Jong fluency), not whether it is *grammatical* — GEC 0.9%, LLM judge 0.0% (NNLS weight 0).
Root cause, confirmed: **Whisper normalises grammar**, laundering the errors before GEC/LLM see them.
That proxy is the OOF ~0.528 wall.

**v5 fix:** add a **verbatim CTC transcript** (`facebook/wav2vec2-large-960h`, greedy — no LM decoder,
so errors survive) and mine the signal Whisper destroys. All additive, all guarded, judged on
in-fold OOF + per-fold std. Only 2 Kaggle runs left, so everything ships in one run and the
NNLS+OOF machinery zero-weights whatever doesn't help.

| # | Lever (CONFIG flag) | Mechanism | Hypothesis on OOF RMSE | Risk |
|---|---------------------|-----------|------------------------|------|
| 1 | `USE_CTC` verbatim CTC transcript | reuses the SSL wav2vec2 model with a CTC head; cheap greedy decode | enables 2–5 | none (self-disables to v4 on failure) |
| 2 | `USE_DIVERGENCE` `dv_*` = Whisper↔CTC edit distance | how much Whisper cleaned up = direct disfluency/error proxy; turns the normalisation bug into a feature | **-0.005 … -0.02 (best EV/effort)** | low |
| 3 | `USE_CTC_LING` `c_*` (LanguageTool err-rate, repeats, MATTR, ppl on verbatim text) | grammar errors survive on CTC text | -0.003 … -0.015 | low |
| 4 | `GEC_ON_CTC` — GEC fed the verbatim transcript | v4's dead 0.9% lever, now fed the input that carries the signal | uncertain: -0.00 … -0.02 | low |
| 5 | `LLM_ON_CTC` — judge fed the verbatim transcript | revives the weight-0 judge; still a 3B model, so uncertain | -0.00 … -0.02 or neutral | low |
| 6 | `ADD_HUBER` LGB-Huber blend member | robust to noisy 769-row MOS labels; diversity for NNLS | -0.002 … -0.01 | low |
| 7 | bagged isotonic calibration (auto vs single, nested-chosen) | lower-variance version of v4's biggest lever (−0.0124) | -0.001 … -0.005 over single | low |

**Safety invariants (verified offline before upload):** all 15 cells compile; a synthetic dry-run
exercised the full §9 assembly → in-fold PCA → 6-member NNLS blend → bagged calibration → 216-row
submission; every new feature function is finite on empty/single-word/disfluent inputs; the honest
calibrated OOF equals the nested best (no in-sample leakage into the reported number). If CTC can't
load, `USE_CTC` self-disables and v5 == v4.

**Caches:** reuses v4's `ac_*`, `asrmeta_*_v4`, `lg_*_v4`, `w2v_*`, `wl_*` verbatim (Whisper + SSL do
not rerun if attached). New artifacts: `ctc_*_v5.json`, `ctcfeat_*_v5`, `gec_*_c_v5`, `llm_*_c_v5` —
never collide with v3/v4. **If ASR/CTC changes, delete the transcript-dependent `_v5`/`_c_v5` caches
before re-running.**

**Decision rule:** keep v5 as finals pick #1 only if BOTH OOF RMSE and per-fold-std hold vs v4
(0.5281 / 0.0315) AND the signal-check corrs print (`dv_edit_rate` and `c_err_rate` vs label) come
out clearly negative. The §5b/§6/§7 correlation prints are the tell: if the divergence corr is
strongly negative, the verbatim path is real; if ~0, Whisper and CTC agree too much on this data and
v5 collapses to a v4-equivalent (still safe). v4 (0.4354) stays as finals pick #2 regardless.

**RUN PLAN (2 runs):** Run #1 = full v5, paste back §9 (per-model + blend OOF + NNLS weights),
the calibration line, the §5b/§6/§7 corr prints, §10 feature-group readout + worst-10, and the LB.
Run #2 = one targeted adjustment based on what the corrs say (see NEXT), or a resubmit.

## v5 MEASURED RESULT (run 2026-09-30) — the verbatim path WORKS

| pipeline | OOF RMSE (raw) | OOF RMSE (calibrated) | Pearson | per-fold std |
|----------|----------------|-----------------------|---------|--------------|
| v4 (prior best) | 0.5420 | 0.5281 | 0.9045 | 0.0315 |
| **v5 (verbatim CTC)** | **0.5328** | **0.5189** | **0.9080** | **0.0255** |

All three moved favorably together (RMSE −0.0092, Pearson +0.0035, std −0.006) -> **trusted win**.
Calibration chose single-iso (−0.0121), bagged was a tie (0.5194). LB: `TBD — submit v5 submission.csv`.

**MECHANISM CONFIRMED — the grammar signal Whisper laundered out is now recoverable:**
- `corr(c_err_rate, label) = -0.623` (verbatim LanguageTool error-rate) — now the **single most
  label-correlated feature in the pipeline**. In v4 this signal was dead (GEC 0.9% on Whisper text).
- `corr(dv_edit_rate, label) = -0.487` (Whisper<->CTC divergence) — cleanup amount tracks grammar.
- Feature-group gain: CTC-verbatim 4.7% + divergence 2.4% (new), ASR-confidence still #1 at 19.3%,
  GEC 1.1%, LLM 0.9%, POS 1.8%, De-Jong 1.5%.

**What worked / what it revealed:**
- **LGB-Huber was the biggest single modelling win**: NNLS weight 0.409 (largest), plain lgb -> 0,
  et -> 0. Robust loss carries the tree signal on the noisy 769-row labels. Keep it.
- **LLM judge revived by CTC text**: standalone Pearson 0.40 (v4) -> 0.55 (v5), but weight 0.005 and
  OOF 2.07 (scale-miscalibrated). A stronger judge (7B) on verbatim text is the clearest next lever.
- **Grammar signal is real but under-weighted (~7%)** — dominated by MFCC-std + ASR-confidence. The
  way up is to AMPLIFY the proven grammar axis, not add more acoustic features.
- Residual error = regression to the mean at the extremes (worst-10: several true-5.0 -> ~3.5),
  part label noise ("simple but grammatical" clips scored 5), part Bayes-optimal RMSE compression.

## NEXT MOVES after v5 (ranked; 1 full run left after submitting v5)
0. **Submit v5's submission.csv now** (free, no re-run) -> get the LB. If v5 LB <= 0.4354 it becomes
   finals pick #1 and v4 drops to pick #2. Decide Run #2 only after seeing the LB.
1. **Amplify the proven grammar axis (safest, cheap):** expand verbatim error features (LanguageTool
   error CATEGORIES + errors-per-sentence + error-free-sentence ratio on CTC text) and add a
   **grammar-stack blend member** — a small model trained ONLY on {c_*, dv_*, gec_*, llm_score} whose
   OOF becomes an explicit NNLS member, surfacing the -0.623 signal instead of burying it under 350
   dims. Zero new heavy models; recompute only ctcfeat_*_v5.
2. **Stronger judge (higher ceiling, higher risk):** Qwen2.5-7B-Instruct 4-bit on the verbatim
   transcript. Judge revived to Pearson 0.55 on 3B -> a 7B could become a real blend member. Guard
   with fallback to 3B so a T4 OOM can't waste the last run.
3. Better CTC (`-large-960h-lv60-self`) for cleaner verbatim text -> cleaner c_err_rate. Low-risk,
   uncertain magnitude; only if bundled.

## v6 — RELIABILITY-FIRST (`build_v6.py` -> `solution_v6.ipynb`); data facts VERIFIED off the CSVs/WAVs

Pivot: instead of a bigger model, make the estimate trustworthy and fix the target/validation. Two
data facts were verified directly (stdlib `wave` + pandas on the kagglehub cache), not assumed:

| Verified fact | Numbers | Consequence |
|---------------|---------|-------------|
| 37 training labels are **0** | files `audio_5037..5073`, contiguous high-ID block, only in `train/`; loud/normalised **RMS~0.122 peak~0.950** vs test RMS 0.055 peak 0.611 | label 0 is OUTSIDE the 1-5 rubric; an anomalous batch; test IDs are 0..215 -> test almost certainly has NO zeros; these 37 unfittable points **inflate the OOF RMSE** |
| **Train/test duration shift** | train-positive median **60.07s**, test median **45.06s**; train 45s clips avg **2.93** vs 60s clips avg **3.56** | random CV over train is NOT test-like; test is 45s-dominant (~56-63%) -> validate test-like |
| predict-mean RMSE | all-769 **1.238**, positive-only **1.014** | the zeros add ~0.10-0.15 of the reported OOF; true 1-5 performance is likely already near the 0.4263 LB |
| filename ID vs label | corr **-0.554** but 100% driven by the zero block; 212 test names collide with train names but are DIFFERENT audio in `test/` | never use filename-ID as a feature; keep keying by directory (pipeline already does) |

Reviewer catch (correct, adopted): v5's §9 fit NNLS weights + calibration on all rows, then scored
the blend on those same rows -> **in-sample optimistic** (~0.003-0.005). Base-model OOF was honest;
the second stage was not.

v6 (all on cached features -> cheap rerun; NO new model, NO re-extraction):
- Compares **all-769 vs 732-positive** training, both scored on the same 732 positive 1-5 targets.
- **Nested outer-CV of the whole stack** (NNLS + isotonic fit inside each outer-train only) -> honest.
- Reports **test-like (duration-reweighted) RMSE** + per-cohort (short/long) + Pearson; picks the
  config by test-like RMSE. Clips submission to **[1,5]** (test has no zeros).
- Tests a **compact grammar specialist** (Huber on error + syntactic-range feats) and a **-CTC**
  ablation; keeps each only if it improves the honest test-like held-out score.
- §8b prints label anomalies, the duration shift, and a wavlm-cosine **speaker-grouping heuristic**
  (if many near-duplicate pairs -> switch to GroupKFold).

Verified offline before upload: all 17 code cells compile; the honest-eval harness (train_stack ->
nested outer-CV -> test-like weighting -> grammar member -> -CTC ablation -> [1,5] submission) passed
a synthetic dry-run with a zero-block + duration shift; splice order confirmed (feature cells = v5).

**Cheap-rerun requirement:** a fresh Kaggle notebook has an empty `/kaggle/working`, so attach the
v5 run's OUTPUT (Add Input -> Notebook Output) — §1b copies the caches in. Otherwise it recomputes
(full slow run). Cheapest: paste v6's §9-§11 into the already-run v5 notebook and re-run from §9.

**Decision rule:** keep the v5 CSV (0.4263) as benchmark. Adopt v6's submission only if a config
beats v5 on the **test-like** honest RMSE (not random-CV). Expect **B pos-732** (drop the zeros) to
win; if `+gram` or `-CTC` don't improve test-like held-out, don't carry them.

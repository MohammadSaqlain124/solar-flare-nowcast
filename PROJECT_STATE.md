# PROJECT_STATE — Solar Flare Nowcasting on Aditya-L1 (TWIN-X)

**Owner:** Saqlain · **Status:** Complete. Full pipeline on real Aditya-L1 data,
three validated outputs (detect / warn / classify), and a live demo frontend.
Headline result: HEL1OS hard X-rays lead the soft peak by ~3.4 min vs GOES's
44 s. Results in sections 6-7c; remaining work in section 12.

---

## 1. The project in one paragraph

A real-time-*capable* solar flare nowcasting system on Aditya-L1 X-ray data. The
differentiator: fuse **SoLEXS** (soft, 2-22 keV) and **HEL1OS** (hard, 20-30 keV
CdTe) as one signal. Soft shows gradual pre-flare heating; hard shows the
impulsive burst - the Neupert effect, hard X-ray ~ d/dt(soft). Most systems use
only soft X-rays or magnetograms. Three outputs: detect a flare, classify it
(B/C/M/X), give a few minutes of early warning before the peak.

**Honest scope:** X-rays give nowcasting + a few minutes of lead, NOT hours-ahead
prediction (needs magnetograms - future work). "Real-time" is architectural;
PRADAN data is archived, so the demo replays archived FITS as a live feed.
Working title: **TWIN-X**. Pitch deck exists.

---

## 2. Development strategy (as executed)

Built the whole pipeline on **GOES XRS** first (public, instant, 1s, HEK flare
catalogue) because PRADAN needed registration. Phases 1-3 are all GOES. Then
registered for PRADAN, downloaded real SoLEXS + HEL1OS, and swapped them in as
Phase 4 - reusing the *exact same* cleaning/featurizing/training code so the
comparison measures the instruments, not the code.

---

## 3. What's built

**Notebooks** (`notebooks/`): `01_explore_goes`, `02_neupert_all_flares`,
`03_cadence_experiment`, `04_cleaning`, `05_windows_and_labels`, `06_split`,
`07_baseline`. Exploration only now - the logic lives in `src/`.

**GOES pipeline** (root): `run_pipeline.py` runs 02->07 for any date range;
`config.py` is the single source of truth for every knob.

**Shared library** (`src/`):
- `metrics.py` - confusion, TSS/HSS/**MCC**/recall/precision, Wilson CI,
  event-level recall, `evaluate()`. skill math byte-identical to notebook 07.
- `operating_point.py` - threshold sweep, PR curve, `pick_threshold` (defaults
  to HSS - the fix for the TSS mirage).
- `losses.py` - weighted BCE + focal, on logits.
- `nets.py` - `CausalConv1d` (left-pad + chomp), `ConvBlock` (dilation +
  optional residual), `FlareCNN` (shared trunk, detection + warning heads).
  cnn = dilation 1 no skip; tcn = dilations [1,2,4,8] + residual.
- `data_torch.py` - torch loaders, accepts custom split masks, per-task
  pos_weights.
- `trainer.py` - `fit_and_eval()`, the shared training core every driver calls.
- `folds.py` - walk-forward fold masks with event integrity (no flare straddles
  a split).
- `clean.py` - notebook 04's cleaning + causal background, factored. also
  `aditya_to_clean()`, the phase 4 adapter.
- `featurize.py` - notebooks 05+06 factored: resample, 6 causal features,
  labels, windows, chronological split, scaler -> `dataset.npz`.
- `aditya_load.py` - PRADAN FITS tree -> fused 1s soft/hard parquet.

**Drivers** (root): `train.py`, `multi_seed.py`, `walk_forward.py`,
`sweep_warn_weight.py` (all take `--arch`, `--warn-weight`, `--data`),
`build_aditya.py` (FITS -> fused parquet), `build_aditya_dataset.py`
(fused parquet -> clean -> featurize -> npz).

**Tests** (`tests/`), all passing:
- `reproduce_baseline.py` - harness reproduces notebook 07's phase 2 table to 4dp
- `reproduce_dataset.py` - `featurize.py` reproduces `dataset.npz` (labels and
  event_id exact, features to ~5e-7)
- `smoke_cnn.py` - model + both losses + backward + tcn path on real data
- `smoke_aditya.py` - loader on synthetic FITS: MJD->unix exact, gaps preserved,
  low-coverage day dropped

**Data** (`data/`, gitignored):
- `dataset.npz` - GOES, 107,173 windows, 117 test events (29 M+)
- `aditya/solexs/`, `aditya/hel1os/` - raw PRADAN FITS trees
- `aditya_clean_1s.parquet` - fused soft/hard, 46 days, 3.97M rows
- `aditya_goesshape_1s.parquet` - same, in GOES clean schema
- `aditya_dataset.npz` - 59,036 windows, 111 test events (29 M+)
- **careful:** `goes_clean_1s.parquet` on disk is only the 7-day dev slice, not
  the 80-day version that built `dataset.npz`. re-run `run_pipeline.py` if you
  need the full clean frame.

**Models** (`models/`, gitignore it): `tcn.pt` (GOES), `tcn_aditya.pt`.

---

## 4. Measured facts (from OUR data - cite, don't re-derive)

Physics:
- **Median flare rise time: 8 min** - physical ceiling on early warning.
- **Derivative lead (soft rise-rate vs soft peak): 2.17 min** (1.72 fast flares).
- **GOES xrsa lead: 44 sec** - GOES's harder channel is useless for warning.
- **HEL1OS hard lead: MEASURED (nb08) = median 3.38 min (202 s)**, 95% CI
  [2.95, 3.85], n=418 detectable flares (98% of measured). ~4.6x GOES's 44 s.
  Model-free physics - the single most quotable number. Full detail in 7b.1.
- **e-folding decay ~12.6 min, class-independent.**
- **Horizon:** 5-min ~98% nowcastable; ~83% at 10 min. HORIZON_MIN = 5.
- **Cadence:** 1s chosen by experiment. store 1s, feed model 10s (180-step
  windows = 30 min).

GOES dataset (80 days, Aug 1 - Oct 20 2024): 770 flares (C 510, M 248, X 12),
~107k windows. Split train 523 / val 100 / test 117 events (29-30 M+). X-class
not evaluable (2 in test). Warning ~88:1 imbalanced (pos_weight ~87.3);
detection near-balanced (~1.1).

Aditya instruments (from the user manuals):
- **SoLEXS** 2-22 keV, 1s cadence, SDD1 + SDD2. **use SDD2** - SDD1 saturated
  during 2024 solar max. LC file: `RATE` ext, `TIME` (unix s) + `COUNTS`.
- **HEL1OS** CdTe 8-70 keV / CZT 20-150 keV. light curves 1s, **five energy
  bands as separate named HDUs**. we use **CdTe1 20-30 keV** - above SoLEXS's
  ceiling, non-thermal, the band GOES cannot see. cols `MJD`, `ISOT`, `CTR`,
  `STAT_ERR`.
- time bases differ: SoLEXS unix seconds, HEL1OS MJD. `unix = (mjd-40587)*86400`.

Aditya coverage (Aug-Oct 2024 window): SoLEXS 77-78 days (near-continuous),
HEL1OS **49 days** (patchy between days, but ~99% complete within a day; six
days at ~50% are single-dump days). Overlap 47, one dropped for low soft
coverage -> **46 fused days, 92.9% both-channel coverage**.

---

## 5. Phase 2 baseline (the numbers everything is measured against)

Threshold on `log_ratio`, chosen on val, tested once:

| Task | TSS | HSS | recall | precision | events |
|---|---|---|---|---|---|
| detection | 0.432 | 0.344 | 0.730 | 0.422 | 117/117 |
| warning | 0.782 | **0.122** | 0.883 | **0.074** | 29/29 |

Warning TSS 0.782 is a **mirage** (precision 0.074, ~1606 false alarms). HSS
0.122 tells the truth. **Report warning by HSS + precision, never TSS.**

---

## 6. Phase 3 results — GOES learned models (DONE)

Shared-trunk TCN, two heads, **warn_weight = 10** (tuned by seed-averaged sweep;
w=10 won on both mean HSS and tightest spread, confirmed at 5 seeds).

**Fixed split, 5 seeds:** detection TSS 0.564 ± 0.021; warning HSS **0.529 ±
0.010**, precision **0.445 ± 0.033**, event recall 94% ± 3%.

**Walk-forward (4 folds, w=10):**
- warning per-fold: TSS 0.437 ± 0.064, HSS **0.450 ± 0.052**, precision 0.476 ±
  0.068
- detection per-fold: TSS 0.197 ± 0.218 (unstable - see findings)
- pooled event recall: detection **93% [91%, 95%]** (n=444), warning
  **79% [72%, 85%]** (n=131)

vs baseline warning HSS 0.122 / precision 0.074 = ~4x HSS, ~6x precision, holding
across fixed split, 5 seeds, and rolling-origin retraining.

---

## 7. Phase 4 results — real Aditya-L1 (DONE)

Same code, different instruments. `aditya_dataset.npz`: 59,036 windows, train
41,955 / val 3,632 / test 13,419, detection positives 43.9%, warning 1.33%,
**111 test events (29 M+)**.

**Fixed split, 5 seeds:** detection TSS 0.328 ± 0.124 (unstable); warning HSS
0.436 ± 0.042, precision **0.506 ± 0.068**, recall 0.404 ± 0.082.

**Walk-forward (4 folds, w=10) - the honest headline:**
- warning per-fold: TSS 0.544 ± 0.092, HSS **0.484 ± 0.049**, precision 0.456 ±
  0.082
- detection per-fold: TSS 0.206 ± 0.086
- pooled event recall: detection **90% [86%, 93%]** (n=302), warning
  **89% [80%, 94%]** (n=87)

### GOES vs Aditya, walk-forward, side by side

| warning | GOES (80 days, soft-only) | Aditya (46 days, soft+hard) |
|---|---|---|
| HSS | 0.450 ± 0.052 | **0.484 ± 0.049** |
| precision | 0.476 ± 0.068 | 0.456 ± 0.082 |
| TSS | 0.437 ± 0.064 | **0.544 ± 0.092** |
| pooled event recall | 79% [72-85%], n=131 | **89% [80-94%], n=87** |

**⚠ SUPERSEDED — RETRACTED, see 7b.2.** The table above compares GOES on 80
days vs Aditya on its 46 - DIFFERENT flares. The paired same-46-days rebuild
(7b.2) showed those 46 days were simply easier: on identical days GOES and
Aditya TIE (GOES recall jumps 79%->91% when restricted to Aditya's dates). Do
NOT claim "Aditya warns better" - that claim is retracted. Honest statement now:
Aditya MATCHES the GOES workhorse on identical days, from one newer instrument,
43% fewer observing days. The differentiator is the physics lead (7b.1).

---

## 7b. Hard-lead physics + fusion interrogation (DONE — this session)

### 7b.1 HEL1OS hard-channel lead (nb08, verified) — THE HEADLINE
- HEL1OS 20-30 keV peaks median **3.38 min (202 s)** before the SoLEXS soft peak,
  95% CI [2.95, 3.85], n=418 detectable flares (98% of measured). vs GOES 44 s =
  ~4.6x. Fast flares (rise<=5 min) still 2.02 min. C 3.22 / M 3.62 min (M>C,
  consistent with stronger non-thermal in bigger flares).
- Anchor: identical measure() on the 1s GOES slice reproduces stored 2.17 min
  derivative / 0.73 min xrsa (got 1.95 / 0.62, in band) -> method sound.
- Robust across smoothing widths (3.2-3.7 min); not a filter artifact. Model-free.
- Lesson: 1-min cadence inflates the derivative lead (2.2->3.0). The GOES anchor
  "FAIL" at 3.0 was that cadence effect, not a bug - run the anchor on the 1s slice.

### 7b.2 Paired same-46-days GOES (build_goes46.py) — RETRACTS "Aditya better"
- Subset the GOES dataset.npz to the exact 46 Aditya dates, walk-forward:
  GOES-46 warning HSS 0.477+/-0.049, recall **91% [84-96] n=93**.
  Aditya-46 warning HSS 0.483+/-0.024, recall **87% [79-93] n=87**. -> TIE.
- GOES recall jumped 79% (80 days) -> 91% (Aditya's 46 days): those days were
  easier. The 89-vs-79 gap was the DAYS, not the instrument. Claim retracted.

### 7b.3 Lead-time per model (lead_time.py)
- First-fire lead before peak: GOES-46 and Aditya-46 BOTH median 3.00 min, IQR
  [2,4], only ~3% of events fire at the 5-min horizon ceiling. The soft rise
  already carries the warning signal within 5 min; the hard channel's 3-min
  physics lead adds NO extra earliness at this horizon.

### 7b.4 Soft-only ablation (build_ablation.py), 3 seeds
- Drop log_xrsa/hardness/rate_a: warning HSS soft-only **0.474+/-0.006** vs fused
  **0.467+/-0.030**; recall ~83% vs ~86% (overlapping Wilson CIs). TIE within
  seed scatter. Soft-only is also the STEADIER model (lower seed variance).
- Hard channel adds nothing to 5-min warning; soft rise carries the full signal.
- NB: quote the seed-averaged fused warning HSS **~0.47**, NOT single-seed 0.483.

### 7b.5 Classification head — C vs M+ (nets/trainer/data_torch/walk_forward + classify_events.py)
- Third head on the shared trunk, masked to active windows (y_class>=1), target
  M+ (y_class>=2), gated by cls_weight (cls_weight=0 recovers det/warn). Adding
  it left warning inside the locked seed band (no regression).
- Per-event (flux-peak anchored) AUC: fused **0.415**, soft **0.468** - BOTH
  ~chance. Head predicts the base rate (calls everything M+). Hard does NOT help.
  Classification is the WEAK third output; ship as best-effort label, say so.
- Aggregation lesson: mean-over-event AUC read 0.30 (inverted) - a bug, not the
  model. M+ flares have long decay tails; averaging over them anti-correlates
  with class. Anchor the per-event score to the FLUX PEAK (argmax soft). Verified
  on synthetic (mean->0.0, flux-peak->1.0).

### 7b.6 The arc (put in the deck)
The non-thermal precursor is REAL (3.4 min, measured) but at the MODEL level the
hard channel is REDUNDANT for short-horizon warning - proved three ways (paired
days, per-model lead, ablation) and classification agrees. Honest headline: "the
precursor is real, but for 5-min warning the soft channel already suffices." A
tested negative result is a contribution, not a gap.

### 7b.7 New files this session
- notebooks/08_hard_lead_aditya.ipynb - hard-lead measurement + GOES anchor.
- build_goes46.py - subset GOES dataset.npz to the 46 Aditya dates.
- lead_time.py - per-model first-fire warning lead.
- build_ablation.py - soft-only (3-feature) dataset.
- classify_events.py - per-event C-vs-M+ readout (flux-peak anchored, ROC-AUC).
- src edits: nets.py (+cls_head, 3 outputs), data_torch.py (y_class in loaders +
  classify pos_weight), trainer.py (masked cls loss, n_feat from X.shape[2],
  cls threshold/eval, cls_prob/cls_soft in test dict), walk_forward.py
  (--cls-weight + classify row), tests/smoke_cnn.py (4-tuple batch, 3 heads).

---

## 7c. Frontend v1 (DONE — this session)
- **frontend/twin-x-demo.html** - single self-contained file (vanilla JS +
  canvas), no build, no server. Opens in any browser; screen-share ready.
- Replays a REPRESENTATIVE M-class flare: animated coronal-loop background that
  tenses/warms as the flare builds, a live soft+hard flux trace, and the three
  outputs (detection / warning / classification). Warning IGNITES ~3 min before
  peak with a live countdown - driven by the SOFT rise (matches the ablation:
  hard channel redundant for warning).
- On-screen honesty label: "representative replay". Timeline built from real
  flare shapes + the measured ~3-min lead; M1.4 peak; warning fires 3.3 min
  before peak. Warm-twilight palette (no dark-neon). Type: Syne / Instrument
  Sans / Spline Sans Mono.
- Swap to real: replace the SOFT/HARD/DET/WARN/CLS arrays in buildTimeline()
  with exported model outputs from a real test event - nothing else changes.
- Lives in frontend/. Deploy via GitHub Pages or Netlify for a live demo link.

---

## 8. Key findings (these ARE results — put them in the deck)

- **Architecture is not the warning lever.** CNN warning HSS ~0.50 ~= TCN ~0.50.
  The win came from learned features + focal loss + HSS threshold selection.
  TCN only helped detection. LSTM/Transformer will probably not move warning.
- **Warning TSS is a mirage under imbalance;** HSS + precision is the honest
  pair. Learned models *lowered* warning TSS while raising HSS 4x and precision
  6x. That single fact is a slide on its own.
- **Pooling is valid for event recall, INVALID for window TSS/HSS/precision.**
  Each fold picks its own threshold; pooled window-TSS is not the mean of
  per-fold TSS (it went *negative* on GOES detection while event recall was
  93%). Pool only event recall + Wilson CI; report window skill per-fold as
  mean ± std.
- **Detection is data-hungry for a stable operating point.** Walk-forward
  detection TSS swings wildly on early folds, then matches the fixed split by
  the last fold. Report detection from fixed-split numbers; cite the
  walk-forward spread as the data-hunger evidence.
- **GPU nondeterminism:** cuDNN conv kernels aren't deterministic under
  `manual_seed`. Always report mean ± std over seeds, never a single run.
- **Aditya's small val block (3,632 windows, 6%) causes the seed instability.**
  46 non-contiguous days put the chronological boundaries awkwardly. Walk-forward
  sidesteps it entirely - which is why the walk-forward numbers are tighter than
  the fixed-split ones and are the ones to quote.
- **Different base rates:** Aditya detection positives 43.9% vs GOES ~31%, so
  detection TSS is NOT directly comparable across the two datasets. Warning
  (1.33% vs 1.12%) is close enough to compare.

---

## 9. Bugs found + fixed (all SILENT — no error, wrong output)

Theme: things that work small break at scale with no exception.
1. `reindex()` on sub-second timestamps -> 100% NaN. GOES 1s stamps carry a
   sub-second offset (`...00.982231`) but `resample("1min")` labels on clean
   boundaries, so *not one* label matches. Fix: `reindex(method="ffill")`, which
   is also the causally correct choice. **This is the single most dangerous
   line in the project** - it recurs anywhere a coarse series maps back to fine.
2. Parquet round-trip drops ns->us -> all labels zero. Fix: force ns + assert.
3. Flare cache keyed by filename -> stale catalogue. Fix: key includes date range.
4. Single 153-day HEK query hangs. Fix: monthly chunks.
5. No download timeout -> infinite hang. Fix: parfive ClientTimeout.
6. Corrupt .nc fails forever. Fix: purge bad files up front.
7. **event_id built from µs ints compared against ns ints** -> every window came
   back quiet, silently. Fix: force `datetime64[ns]` before the int64 cast.
   Caught only because `reproduce_dataset.py` compared against the stored npz.
8. **`WinError 32`** on Windows: `np.load` on an npz keeps a handle open, so
   `os.unlink` fails. Fix: load inside `with`, best-effort cleanup.
9. **Single-channel coverage check let a broken day through:** Oct 13 had 99%
   hard but 13% soft. Fusion needs both. Fix: `min_soft_cov` guard too.
10. `--data` didn't exist on the drivers, so a "Phase 4" training run silently
    trained on the GOES npz again. Fix: `--data` flag + print the loaded file
    and its test-event count as line one.

**Rules:** any cache keyed on a name needs its params in the key; any timestamp
join needs an assertion it matched something; any coverage guard must check
*every* channel the downstream needs.

---

## 10. Reference recipes (verified exact — do not re-derive)

**Causal background** (`clean.causal_background`, reproduces the stored GOES
`bg_causal` with **0.0 error**):
```
minute = xrsb_clean[xrsb_valid].resample("1min").median()
bg     = minute.rolling("4h", min_periods=10).quantile(0.05)
bg_causal = bg.dropna().reindex(index, method="ffill")
```
It is computed on a **1-minute median series**, not the raw 1s data. A rolling
quantile straight on 1s data matches only ~21% of it.

**Six features:** `log_xrsb, log_xrsa, log_ratio, hardness, rate_smooth, rate_a`.
All causal (backward diffs, trailing rolling). Never `np.gradient`.

**HORIZON_MIN = 5**, from `config.py`. Notebook 05 hardcodes a stale `10` - the
real `dataset.npz` used 5 (`meta[3]`, warning positive rate 1.1%). **Fix that
stale value in the notebook.**

**Classification (C vs M+):** mask to active windows (y_class>=1), target M+
(y_class>=2), pos_weight = C/M+ on active train windows. Per-event score = model
M+ prob at the FLUX-PEAK window (argmax soft), scored threshold-free by ROC-AUC.
NEVER mean over the event - the decay tail inverts it.

**Aditya -> GOES schema:** soft -> `xrsb`, hard -> `xrsa`. Units differ (counts
vs W/m²) but every feature is log'd then z-scored on train, so absolute scale
washes out - that's *why* the comparison is fair.

**Evaluation rules:** metrics are TSS, HSS, MCC, precision, recall - never
accuracy. Chronological split only. Event-level is the honest sample size; quote
the event count and a Wilson CI beside every metric. Scaler fit on train only.
Select on val warning average-precision (threshold-free); pick operating points
on val - detection by TSS, warning by HSS.

---

## 11. Environment notes

- **Train on Colab (GPU), not locally.** The Windows torch install is CPU-only;
  a full `train.py` run took ~1 hour locally vs ~5 min on a T4. Check the first
  printed line says `device cuda`.
- Windows uses PowerShell: `Expand-Archive`, not `unzip`. Paths under
  `E:\Projects\solar-flare-nowcast`.
- PRADAN bulk downloads come as a **Python downloader script**, not the data -
  run it from inside the target folder and it downloads there.
- The PRADAN FITS trees nest differently per instrument and HEL1OS repeats the
  date dirs. **Always discover files by `rglob` + filename date, never build
  paths.**

---

## 12. NEXT (in priority order)

DONE: full backend (three outputs, honestly characterised) [7b.*] + frontend v1
[7c]. **The project is functionally complete.** Remaining, in order:

1. **Deploy the demo** (GitHub Pages: Settings->Pages->main/root, or Netlify
   drop) for a live link - a real asset for the pitch/interview.
2. **Deck refresh:** lead with the 3.4-min physics headline; state the model as
   "matches the GOES workhorse from one newer instrument, 43% fewer days";
   present the fusion-redundant finding as a TESTED strength, not a gap.
3. **Optional / future:** swap real model outputs into the demo; longer horizon
   (8-10 min) where the hard lead MIGHT pay off (coverage drops to ~83%); better
   classifier (weakest output); React port of the frontend for the MERN stack.

Framework: PyTorch, train on Colab (free GPU), VS Code home base. Frontend:
vanilla JS + canvas (single file), portable to React.

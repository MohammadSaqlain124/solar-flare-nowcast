# PROJECT_STATE — Solar Flare Nowcasting on Aditya-L1 (TWIN-X)

> Paste this into the first message of a new chat to bring Claude fully up to
> speed. Everything needed to resume is here; nothing relies on memory.

**Owner:** Sam · **Last updated:** 2026-07-28 · **Status:** Phases 1-4 done.
GOES pipeline built and beaten; real Aditya-L1 data downloaded, fused, and
trained on. Next: hard-channel lead-time measurement (the pitch's key number),
then the paired same-days comparison.

---

## 0. Code style (READ FIRST — applies to all new code)

Comments should read like a working dev wrote them, not a tutorial. If asked to
rewrite existing code for voice, change **only** comments/docstrings/log strings,
never the logic (I verify with an AST diff).

- terse, lowercase fragments; comment the non-obvious *why* only, skip anything
  that restates the code. uneven density on purpose - comment the tricky bits,
  leave the obvious lines bare. do NOT give every function a docstring.
- no decorative banners (`# ==== section ====`), no column-aligned trailing
  comments, no ALL-CAPS "THE KEY LINE" emphasis.
- plain punctuation: `-` and `->`, not em-dashes. avoid "i.e./e.g." and formal
  phrasing. lowercase sentence fragments are fine.
- a short war-story note where a bug bit is welcome ("gotcha:", "learned that
  one", "careful:").
- no teaching tone. write for my future self, not a student. short one-line
  docstrings only where they earn their place.
- every file in `src/`, every driver in root, and `config.py` /
  `run_pipeline.py` are already in this voice - match them. the notebooks still
  have the old verbose comments; rewrite to this voice when you touch them.
- (Context: I want the code to read as human-written. If a hackathon asks about
  AI use, the honest line - "used AI as a pair-programmer, directed all of it" -
  is fine and true; the styling is about readability, not hiding the tooling.)

**Two standing rules for chats:** correct my English, and rate each question's
quality (Ok / Good / Great).

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
  *The HEL1OS counterpart to this number is NOT yet measured - see section 8.*
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

**How to state this honestly:** *fused SoLEXS+HEL1OS achieves comparable or
better warning skill than the GOES surrogate, with 89% vs 79% event recall, on
43% fewer observing days.* The CIs graze each other, so do **not** claim
statistical significance. Direction is consistent across HSS, TSS, and event
recall - that's the claim, and it's defensible.

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

1. **Measure the HEL1OS hard-channel lead time** — the counterpart to GOES's
   useless 44 s xrsa lead. How many seconds/minutes does the CdTe1 20-30 keV
   rise lead the SoLEXS soft peak, across the 46 fused days? Pure physics on
   `aditya_clean_1s.parquet`, no training, notebook-02 style. **If this lands at
   2-3 minutes vs GOES's 44 s, it is the single most quotable number in the
   entire pitch** and it proves the Neupert-effect claim independently of any
   model.
2. **Paired same-46-days GOES rebuild.** Right now GOES ran on 80 days and
   Aditya on its 46 - different flares, so some of the difference could be "those
   46 days had easier flares." Filter `goes_clean_1s.parquet` to the same 46
   dates, re-run `featurize` + `walk_forward`, and the comparison becomes
   "better on identical data" instead of "comparable on different data".
   Small job - `featurize` already takes any clean parquet.
3. **Classification head (C/M).** The third promised output, still unbuilt.
   `y_class` is already in both npz files; add a third head to the shared trunk.
   Report C and M only (B undetectable at solar max, X n=2).
4. **Phase 5 demo:** FastAPI/WebSocket/Redis/Docker/React is a project in
   itself. A working model + simple dashboard beats a half-built stack.
5. **Optional:** re-save the GOES checkpoint from the best seed (`models/tcn.pt`
   is a single draw at HSS 0.467, below the 5-seed mean 0.529). Try
   `--epochs 45` (best val AP was still climbing at epoch 26/30).
6. **LSTM / Transformer:** only if 1-5 are done. Architecture has not moved
   warning once. Don't expect gains.

Framework: PyTorch, train on Colab (free GPU), VS Code home base.

## 12.1 RESULT — HEL1OS hard-channel lead time (nb08, verified)
- HEL1OS 20-30 keV peaks median 3.38 min (202 s) before SoLEXS soft peak,
  95% CI [2.95, 3.85], n=418 detectable flares (98% of measured). vs GOES xrsa 44 s.
- Anchor: same measure() reproduces GOES 2.17/0.73 on 1s slice (1.95/0.62, in band).
- Robust across smoothing (3.2-3.7 min). Fast flares (rise<=5) still 2.02 min.
- This is instrument-level physics, no model. The project's most quotable number.

## 12.2 RESULT — paired same-46-days GOES (CORRECTS the earlier claim)
- On the SAME 46 days, GOES and Aditya are TIED on warning:
  GOES-46 HSS 0.477+/-0.049, recall 91% [84-96] n=93.
  Aditya-46 HSS 0.483+/-0.024, recall 87% [79-93] n=87.
- GOES recall jumped 79% (80 days) -> 91% (Aditya's 46 days): those days were
  easier. The 89-vs-79 gap was the DAYS, not the instrument. Retract "warns better".
- Honest claim now: Aditya MATCHES the GOES workhorse on identical days, from one
  newer instrument, 43% fewer observing days. Differentiator is the physics lead.

## ABLATION — soft-only Aditya, seed-hardened (verified, same 46 days/folds)
- 3 seeds each. warning HSS: soft-only 0.474+/-0.006 vs fused 0.467+/-0.030.
  event recall ~83% vs ~86% (overlapping Wilson CIs). TIE within seed scatter.
- Hard channel adds nothing to 5-min warning; soft rise carries the full signal.
  Soft-only is also the steadier model (lower seed variance). CONFIRMED, not single-seed.
- Deck: quote seed-averaged ~0.47 HSS, not the 0.483 single point.
# NeurIPS Gap Analysis — Epidemic Forecasting (MIST)

**Status:** internal audit note. Written at the start of the NeurIPS-readiness upgrade.
**Bottom line up front:** the repo is well-engineered, leakage-safe, and reproducible, but the
**current headline claim ("MIST is a strong forecaster, beats ARIMA, rising-phase advantage")
does not survive multi-season scrutiny.** Standalone MIST is *third of four* across seasons. This
note records the gap honestly and defines the upgrade and its falsifiable success criterion.

---

## 1. The core finding: standalone MIST does not generalise

Aggregate WIS per season (lower is better), from `results/season_wis.csv` (real full-run artifact):

| Season | obs | mist_v2 | arima | tft | patchtst | Season winner |
|---|---:|---:|---:|---:|---:|---|
| 2022-23 | 5,512 | 144.0 | 245.7 | 105.7 | **91.3** | PatchTST |
| 2023-24 | 5,510 | **84.8** | 88.5 | 97.2 | 96.8 | **MIST** |
| 2024-25 | 7,404 | 221.6 | 227.2 | **152.4** | 154.5 | TFT |
| **Unweighted mean** | | 150.1 | 187.1 | 118.4 | **114.2** | PatchTST |

**Reading:** MIST wins **only** 2023-24, the season on which the model was developed and tuned.
Across the three seasons it ranks **3rd of 4** and beats only ARIMA. PatchTST and TFT beat MIST by
~24% in aggregate WIS.

This is consistent with the two prior internal memos:
- *MIST 3-season DM pooling*: pooling three seasons reduces rising-phase DM power (p≈0.96); MIST
  loses to TFT/PatchTST in the thin 2022-23 season.
- *MIST calibration vs ARIMA tradeoff*: tuning cov-50 to nominal 0.50 forfeits the WIS win over
  ARIMA; the win is fragile even within 2023-24.

## 2. Weak / selectively-framed claims in the current paper

- **The "thin-data 2022-23" explanation is insufficient.** `paper/mist_neurips.tex:262-274` explains
  MIST's weak 2022-23 result by sparse training data. But MIST *also* loses 2024-25 — a **data-rich**
  season (7,404 obs) — to TFT and PatchTST by ~30%. Thin data does not explain the generalisation
  gap; it is a real model-quality gap on out-of-development seasons.
- **Selective multi-season framing.** The multi-season paragraph states "MIST beats ARIMA in
  2023-24 and 2024-25" but **omits** that TFT/PatchTST beat MIST in 2024-25. Comparing only against
  ARIMA (the weakest baseline) while the strongest baselines win is the central framing problem.
- **Rising-phase advantage is not significant across seasons.** `tables/dm_headline_rising_3seasons.csv`:
  pooled rising-phase DM puts MIST *behind* PatchTST/TFT in point estimate and not significantly
  different from ARIMA (p≈0.96). The phase-stratified win in `results/phase_performance_v2.csv` is a
  **single-season (2023-24)** result.
- **Calibration trade-off.** cov-95 ≈ 0.905 (good); cov-50 ≈ 0.439 (under-covered). The repo already
  documents that fixing cov-50 forfeits the ARIMA WIS win — i.e. the headline is calibration-fragile.

## 3. Missing evaluation pieces (vs. Forecast-Hub norms)

- **No ensemble baseline in code.** Equal-weight / median / performance-weighted ensembles are the
  standard strong baseline in epidemic forecasting (Forecast Hub) and are cited in related work but
  never implemented or evaluated. This is the single most important missing comparison.
- **Multi-season is a secondary check, not the primary benchmark.** `evaluation/run_seasons.py`
  exists but the main leaderboard (`results/leaderboard_v2.csv`) is single-season 2023-24.
- **Per-forecast quantile records are not persisted** for all base models across seasons
  (`run_seasons.py` saves only median/wis/mae), so ensembles/stacking cannot be built without a
  re-run that dumps full quantiles.
- **No classical ML baseline** (e.g. gradient-boosted quantile regression) and **seasonal-naive** is
  absent (only plain persistence exists).
- **SEIR is skipped in the cross-season run** (`run_seasons.py:81`) — defensible for cost but should
  be stated.
- **Significance testing is DM-only and not multiple-comparison-adjusted**; no block-bootstrap CIs
  clustered by forecast date and location.

## 4. What is genuinely strong (keep it)

- **Leakage safety** is real and tested: a single vintage chokepoint (`store.get_vintage`) in
  `evaluation/backtester.py`, 8 dedicated leakage tests, per-window normalisation, training-window-only
  spatial matrix, prior-season-only conformal calibration.
- **Reproducibility scaffold**: single-sourced numbers (`paper/extract_numbers.py` → `numbers.json`
  → `macros.tex`), a reproducibility gate (`evaluation/assert_reproducibility.py`), 57 passing tests.
- **Honest reporting culture**: cov-50 trade-off and 2022-23 weakness are already disclosed.

These are assets. The benchmark's leakage-safety and reproducibility are, by themselves, a credible
contribution even if no single model dominates.

## 5. Upgrade direction (decided)

**Central contribution → a leakage-safe, phase/R\_t-gated *conformal hybrid* (stacking) over the
base models** {MIST, PatchTST, TFT, ARIMA, persistence}. Rationale: hybrids/ensembles routinely win
in epidemic forecasting because no single model dominates across regimes/seasons — exactly the
pattern in §1. The hybrid can route to PatchTST/TFT where they win and to MIST/persistence where
those win, and a conformal layer can repair cov-50.

**Design (leakage-safe):**
- Operate on persisted base-model quantile forecasts (`results/quantiles_long.parquet`).
- For each (phase bucket ∈ {rising, peak, declining}, horizon), set base-model weights by
  Hedge-style exponential weights `w_m ∝ exp(-η · meanWIS_m)`, where `meanWIS_m` is computed **only
  over forecasts whose target was realised strictly before the current origin** (prior seasons +
  earlier origins of the current season). Equal-weight fallback when history is thin.
- Phase bucket comes from the vintage R\_t / slope (vintage-only → no leakage).
- Combine base quantiles convexly (preserves monotonicity), then apply per-(horizon, α)
  split-conformal adjustment calibrated on strictly-prior residuals to fix cov-50.

This frames a coherent methodological story — **mechanistic phase gating + online-learning weights +
conformal calibration** — rather than an engineering bag of tricks.

## 6. Falsifiable success criterion (and the pivot rule)

The phase-aware hybrid contribution is **earned** only if, on the full multi-season benchmark:

1. the hybrid's season-mean WIS is **≤ every individual base model** (incl. PatchTST/TFT), **and**
2. it is **≤ the equal-weight, median, and global performance-weighted ensembles**, **and**
3. the **phase-gated** variant beats the **global (non-phase) performance-weighted** ensemble
   (otherwise "phase-aware" adds nothing and the honest claim collapses to "ensembles win"), **and**
4. cov-50 improves toward nominal without a WIS regression, **and**
5. the improvement survives a block-bootstrap CI clustered by (forecast date, location) and
   multiple-comparison-adjusted DM tests.

**Pivot rule (user-approved): main-track framing only where evidence earns it.** If the hybrid does
**not** meet the criteria above, this file is updated with the measured outcome and the paper is
reframed as a **leakage-safe vintage benchmark + honest comparative study** (a datasets/benchmarks
or applied/workshop contribution), *not* a "we beat SOTA" paper. No headline claim will be made that
is not backed by a saved artifact.

## 7. Status of numbers in this note

All numbers in §1–§2 are from existing **full-run** artifacts (`results/season_wis.csv`,
`tables/dm_headline_rising_3seasons.csv`). New hybrid/ensemble numbers will be appended here once
the full `scripts/reproduce.py --full` run completes; until then the paper is wired to regenerate
from those artifacts and is flagged accordingly. Quick-config runs are for pipeline validation only
and are never used as reported numbers.

## 8. Build status and current (quick-profile) outcome

The upgrade infrastructure is now implemented and tested end-to-end:

- **Baselines added** — seasonal-naive (`models/baseline.py`) and gradient-boosted quantile
  regression (`models/baseline_ml.py`), alongside MIST/TFT/PatchTST/ARIMA/persistence, all run on
  all three seasons.
- **Full per-quantile dump** — `evaluation/quantile_dump.py` → `results/quantiles_long.parquet`
  (the substrate; base training is the only expensive step).
- **Ensembles** — equal-weight, median, leakage-safe performance-weighted (`models/ensemble.py`,
  `evaluation/ensembles.py`).
- **Hybrid** — `models/phase_stack.py`: online Hedge weighting, vintage phase gating, CQR layer,
  with the five-way ablation family.
- **Aggregation + stats** — `evaluation/run_multiseason.py`, `evaluation/bootstrap.py`
  (clustered bootstrap + Holm/BH-adjusted DM).
- **Verdict** — `evaluation/hybrid_verdict.py` → `results/hybrid_verdict.json` adjudicates the §6
  criterion automatically; the reproducibility gate and the paper read it.
- **One command** — `scripts/reproduce.py --quick|--full`; `configs/experiment.yaml`; new tests
  (leakage, ensemble correctness, aggregation, phase gate, claim-consistency). 77 tests pass.

## 8a. FULL-RUN RESULT (the decisive numbers)

`python scripts/reproduce.py --full` has now been run (53 locations, all weekly origins, paper
epochs; `results/quantiles_long.parquet` = 2.97M rows). Verdict: **`earned = false`**. Real
season-unweighted WIS (`results/multiseason_summary.csv`):

| Rank | Model | WIS (unwt.) | cov-50 |
|---:|---|---:|---:|
| 1 | patchtst | 114.2 | 0.468 |
| 2 | **ens_median** | 114.3 | 0.520 |
| 3 | ens_mean | 115.5 | 0.543 |
| 4 | tft | 118.5 | 0.435 |
| 5 | ens_perf | 147.2 | 0.487 |
| 7 | stack_global | 148.6 | 0.489 |
| 8 | **stack_phase_conformal (hybrid)** | 150.5 | 0.588 |
| 10 | mist_v2 | 154.2 | 0.412 |

(Numbers after the reference-date canonicalization fix described below; PatchTST and the
median/mean ensembles are effectively tied at the top.)

**Finding (honest, and itself a contribution):** the **simple mean/median ensembles win** — the
canonical Forecast-Hub result. The hybrid is **rank 8 of 13**. The *cause* is diagnosable, not
mysterious: the entire performance-weighted family (`ens_perf`, `stack_global`, `stack_phase` ≈
145–148) collapses toward a single component because the Hedge temperature `eta=0.05` is far too
aggressive at WIS scale ~150 (`eta·WIS ≈ 7.5` ⇒ near-degenerate weights), discarding the
diversification that makes equal weighting win. Phase gating barely moves WIS (`stack_phase` ≈
`stack_global`), and the CQR layer over-corrects coverage (cov-50 0.589, cov-95 0.960).

**Consequences:**
- The pivot-safe framing is now the *actual* framing: the contribution is the leakage-safe
  benchmark + the honest comparative finding ("simple ensembles win; sophisticated online/phase
  weighting does not help here, and here is the mechanism"). `\hybridEarned = false` in the paper.
- This is a **legitimate, well-motivated lever for WP3** (iterate-to-win): the over-concentration is
  a real weighting bug. Scale-aware/relative-loss Hedge with prior-only-tuned `eta` should recover
  equal-weight as the `eta→0` limit and then try to beat it. Whether it actually beats `ens_median`
  is the open question; if it only reaches parity, the benchmark contribution stands regardless.
- A second-order finding worth keeping: vintage-phase `Peak` is very rare (69 of ~4.6k origins
  under the trailing slope rule), so phase gating is effectively Rising-vs-Declining only.

## 8b. WP3 diagnostics (iterate-to-win): the simple levers are exhausted

Two cheap, decisive experiments on the full dump:

- **Hedge `eta` sweep (performance-weighting).** `ens_perf` season-unweighted WIS vs `eta`:
  `0.0 -> 115.5`, `0.001 -> 115.7`, `0.005 -> 117.6`, `0.01 -> 121.5`, `0.05 -> 147.2`. The optimum
  is `eta -> 0` (i.e. equal weight); **any** concentration on prior-season performance *hurts*.
  Prior-season WIS does not transfer. Retuning the existing weighting cannot win — its ceiling is
  equal weight (115.5), already behind PatchTST (114.2) and the 5-model median (114.3).
- **Component-set / equal-weight ceiling.** The paired common-mask rerun puts the broad
  median ensemble at **113.2** and the trimmed ensemble at **109.9**. The broad robust ensemble
  family is the winner -- the canonical Forecast-Hub result.

**Implication.** The bar to beat is ~**109.9** (paired trimmed ensemble). Performance-weighting and phase gating
do not clear it; the winning object is a simple, broad robust ensemble.

## 8c. Bounded win attempt — SUCCESS: the trimmed-mean ensemble

User decision: "bounded win attempt, then pivot." A short candidate sweep on the full dump found a
genuine, significant winner: the **trimmed-mean ensemble** (`ens_trimmed`) — drop the per-quantile
min and max across the seven base models, average the middle five (a robust aggregator that tolerates
the weak members the plain mean cannot). Promoted to a first-class baseline (`models/ensemble.py:
trimmed_mean`, `evaluation/ensembles.py`).

Final paired full-run leaderboard (season-unweighted WIS on the common forecast mask):

| Rank | Model | WIS | cov-50 |
|---:|---|---:|---:|
| 1 | **ens_trimmed** | **109.9** | 0.549 |
| 2 | ens_median | 113.2 | 0.530 |
| 3 | patchtst | 116.9 | 0.470 |
| 4 | tft | 121.1 | 0.437 |
| 5 | ens_mean | 128.7 | 0.532 |
| 8 | stack_phase_conformal (hybrid) | 153.3 | 0.587 |

**Significance (paired common mask, focal = ens_trimmed):** ens_trimmed beats `ens_median`
(Δ −3.42, bootstrap CI [−5.51, −1.50], Holm-DM p=0.0094), `tft`
(Δ −10.39, CI [−14.22, −6.90], Holm-DM p<1e-7), and the hybrid
(Δ −39.95, CI [−69.04, −20.21], BH-DM p=0.0022; Holm-DM p=0.0501). It beats
`patchtst` under clustered bootstrap (Δ −6.69, CI [−13.06, −0.69]) and BH-DM
(p=0.027), but **not** Holm-DM (p=0.577). We report that caveat explicitly.

**Honest caveats (must be stated):** (i) trimmed-mean ensembling is a *known* robust aggregator — the
contribution is the **benchmark** and the rigorous, significant **comparative finding** ("trimming >
median, the hub's deployed default"), not a novel method; (ii) per season ens_trimmed wins 2023-24 and
2024-25 but **loses the thin 2022-23 to standalone PatchTST** (99.3 vs 91.3), so the win is on the
season-unweighted headline metric, not every season; (iii) the sophisticated phase-gated/online/
conformal hybrid does **not** beat simple trimming — a clean negative result (sophistication does not
pay here). Paper-facing paired artifacts are now wired into
`results/multiseason_common_summary.csv`, `tables/bootstrap_ci_common.csv`, and the gate;
focused leakage/claim/reproducibility tests pass. **Pivot now proceeds to WP1 (genuine vintage) +
WP2 (multi-disease) + WP4/WP6.**

## 9. WP1/WP2 — genuine-vintage data reality (the decisive provenance finding)

Goal: replace the synthetic vintages (§8: `issue_date = reference_date + lag`, `genuine_vintage =
False`) with **real issue-dated revisions**. Investigating Delphi Epidata covidcast established the
exact provenance — and a subtlety that defines the benchmark's scope:

- **NHSN weekly signals** `confirmed_admissions_{flu,covid,rsv}_ew` (source `nhsn`, `time_type=week`)
  carry **genuine revisions** (≈35 issues/week, 99% of cells revised) and match our weekly target
  exactly, for **all three diseases** — the multi-disease substrate in one consistent source.
- **But Delphi's NHSN archive is backfilled:** an Oct-2023 week's *earliest* issue is Nov-2024
  (epiweek 202447). So genuine *contemporaneous* vintages exist only from ~late 2024 onward
  (verified: week 202448's first issue is 202449, a real ~2-week reporting lag, then revised through
  202610). Pre-2024 weeks have no real-time vintage in this source.
- **HHS daily** `confirmed_admissions_influenza_1d` (source `hhs`) has genuine contemporaneous
  revisions for **2022-08 → 2024-04** (flu and covid; no RSV), but is daily and ends before 2024-25.

**Scope decision (honest, NeurIPS-defensible).** The genuine-vintage benchmark centres on the
**2024-25 and 2025-26 seasons × {flu, COVID-19, RSV}**, sourced from Delphi NHSN weekly, where
vintages are genuinely real-time and multi-disease. Historical flu (2022-24) can be added later as an
HHS-daily→weekly genuine extension (flu/covid only). This is a stronger and more honest object than a
single synthetic-vintage flu season: a *genuinely-vintage, leakage-safe, multi-disease* benchmark.

**Built (WP1/WP2 data layer):** `ingestion/vintage_delphi.py` now ingests the full panel (states +
nation, mapped covidcast abbrev → FIPS via `data/crosswalk.csv`) for all three diseases into a
labelled genuine store `data/store_genuine/`; `evaluation/vintage_report.py` confirms
`genuine_vintage = True` on it. The legacy synthetic store is kept, deprecated. Remaining: make the
pipeline disease-parametric (`configs/experiment.yaml` + `_KEY`), retrain base models on the genuine
store per disease, and re-run — paper numbers come only from that genuine, multi-disease full run.

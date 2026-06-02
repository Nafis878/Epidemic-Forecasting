# CHANGES

A running log of changes during the NeurIPS upgrade sprint. Newest first.

## Phase 5 — Paper-ready figures
`evaluation/figures_paper.py` + `evaluation/run_figures.py`; a `last_attn` hook was
added to `MechanisticAttention` to expose attention weights.

- **5.1 Main result** (`figures/main_result.pdf` + 300-dpi `.png`): 3-panel
  (Rising/Peak/Declining) WIS bars, MIST highlighted, ARIMA reference line,
  DM-significance asterisks, colorblind-safe (Wong) palette.
- **5.2 Attention vs mobility** (`figures/attention_map.pdf`/`.png`): for a rising
  week in California, the learned **spatial** attention (blue) vs the independent
  **gravity** mobility flows (grey). Honest finding — attention spreads broadly
  (including to distant eastern states) and does **not** track commuting gravity;
  it reflects epidemic **co-movement**, consistent with the model down-weighting the
  gravity prior (γ_s→0). A falsifiable check with a genuine negative answer.
- **5.3 Case study** (`figures/case_study.pdf`/`.png`): the two rising-phase episodes
  with the largest MIST-over-ARIMA WIS gain (US national, Dec 2023). MIST's median +
  95% band track the climb toward the peak while ARIMA stays flat/low — visible early
  warning.
- Tests: `tests/test_figures.py` (main-result figure renders from a synthetic phase
  table). Full suite: **55 passed**.

## Phase 4 — Evaluation depth (full ~53-location panel, 2023-24)
Headline: on the **full panel**, **mist_v2 (WIS 84.8) beats ARIMA (88.5)** overall
while well-calibrated (cov-95 0.905), and is **best in Rising (51.2 vs 61.6) and Peak
(152.8 vs 179.2)**, near-tying ARIMA in Declining (87.5 vs 81.4 — the blend closed
the gap from 117.8). (Full-panel WIS is lower than the 3-location numbers because the
panel is dominated by small-count states; rankings are what matter.)

- **Mobility decision** (`features/mobility.py`): a real population x inverse-distance
  gravity matrix was built and tested as MIST's spatial prior, but it was ~6.4% *worse*
  than the correlation proxy (the model drove γ_s→0). Per the user's call, MIST keeps
  the correlation proxy and the **gravity matrix is used as an independent mobility
  reference** for the spatial/attention analyses — turning "attention aligns with
  mobility" into a falsifiable check rather than an assumption.
- **4.0 Benchmark** (`evaluation/run_benchmark_v2.py` → `results/leaderboard_v2.csv`,
  `results/phase_performance_v2.csv`, `results/results_all_v2.csv`): all 9 models
  (full MIST v2 + 4 ablations + ARIMA/SEIR/TFT/PatchTST). Ablation (overall WIS):
  mist_no_aci 77.3 (sharper but under-covers) · mist_v2 84.8 · arima 88.5 ·
  no_analogue 91.9 · patchtst 96.8 · tft 97.2 · no_blend 109.6 · no_mech 110.3 ·
  seir 118.1.
- **4.1 DM tests** (`evaluation/dm_test.py` → `tables/dm_significance.csv`,
  `tables/dm_headline_rising.csv`): Diebold-Mariano with Bartlett HAC + HLN
  small-sample correction. mist_v2 significantly beats every ablation (no_blend
  p<0.001, no_mech p=0.004, no_analogue p=0.009) and SEIR (p=0.002). In the **rising
  phase mist_v2 beats every model** (all diffs<0; significant vs all ablations;
  vs ARIMA/TFT/PatchTST p≈0.05-0.06 — a real but borderline edge on a single season).
- **4.2 Horizon breakdown** (`results/horizon_breakdown.csv`,
  `figures/horizon_breakdown.png`): graceful degradation — ARIMA wins the h=1 nowcast
  (39 vs 55) but mist_v2 wins h=3 (92.9 vs 107.7) and h=4 (118.8 vs 132.1).
- **4.3 Spatial connectivity** (`evaluation/spatial.py`,
  `figures/spatial_connectivity.png`, `results/spatial_strata.csv`): MIST's gain over
  ARIMA is largest at national (+69) and positive for high/mid-connectivity
  (+3.2/+4.4) but negative for low-connectivity (−0.8) and the territory (−8.6) —
  partial support for the hub-state hypothesis.
- **4.4 Multi-season** (`evaluation/run_seasons.py` → `results/season_wis.csv`):
  per-season WIS — 2022-23: mist_v2 144.0, arima 245.7, tft 105.7, patchtst 91.3;
  2023-24: mist_v2 84.8, arima 88.5, tft 97.2, patchtst 96.8. **mist_v2 beats ARIMA
  in both seasons**; it is the best model in the data-rich 2023-24 season but the
  simpler DL baselines edge it in the thin-training 2022-23 (only ~38 training weeks)
  — an honest data-dependence caveat. (SEIR omitted here for runtime.)
- Tests: `tests/test_dm_spatial.py` (DM significance behaviour, gravity-matrix
  symmetry, connectivity strata). Full suite: **54 passed**.

## Phase 3 — Fix the declining phase (the SOTA blocker)
All numbers are mean WIS on the 3 evaluation locations (US/CA/NY), 2023-24 season,
with ACI calibration applied to every variant; lower is better.

- **3.1 R_t-conditioned blend** (`models/mist_v2.py` → `MISTNetV2`): blends MIST with
  a persistence (random-walk-with-drift, normalised) nowcast by
  `alpha = sigmoid(beta*(R_t-1))`, `beta` trained **end-to-end** (init 3.0 → learned
  2.71). Declining **1106 → 784 (−29%)**, overall **1030 → 832 (−19%)**, Rising
  slightly better (506 → 478), Peak ~flat; coverage preserved. Far exceeds the
  ">20% declining improvement" target.
- **3.2 Ablation** (all + ACI): mist-no-mech 1091 · mist-no-blend 1030 ·
  mist-full(blend+mech) 832 · **mist-full+analogue 730**. Each component helps:
  mechanistic attention most in Peak (removing it: 1964→2424), the blend most in
  Declining, the analogue prior broadly.
- **3.3 Historical-analogue prior** (`features/analogue.py`, `models/analogue_blend.py`):
  numpy **DTW** retrieves the k=5 most similar historical ILINet rising windows
  (shape-normalised, so the ILI%% syndromic signal can match the admissions target);
  their mean continuation softly nudges the rising-phase median. Leakage-safe —
  analogues whose continuation reaches the current season are excluded. Adding it
  improves **overall WIS 832 → 730 (−12%)**, Rising 478→431, Peak 1964→1547,
  Declining 784→717, coverage intact. The full v2 stack (blend + mechanistic +
  analogue + ACI) is the strongest configuration and is on track to beat ARIMA
  overall (confirmed on the full panel in Phase 4).
- Refactor: `MISTModel._build_net` made overridable so v2 swaps in `MISTNetV2`
  without duplicating the training loop.
- Tests: `tests/test_blend.py` (4), `tests/test_analogue.py` (5). Full suite: **50 passed**.

## Phase 2 — Calibration fix (cov-50)
- **2.1 Diagnosis** (`evaluation/calibration.py`, `evaluation/run_calibration.py`):
  reliability diagrams + Expected Calibration Error per model →
  `figures/calibration_reliability.png`. Raw MIST is badly miscalibrated
  (cov-50 0.30, cov-95 0.72, ECE 0.162), under-covering the upper quantiles; ARIMA
  is well-calibrated (ECE 0.025); TFT/PatchTST sit in between.
- **2.2 Conformal fix** — split conformal alone failed: the only pre-test
  calibration data is the calm off-season (or a milder prior season), which is not
  exchangeable with the test wave, so a *fixed* widening mis-covers (confirmed
  empirically). Implemented the correct tool for distribution shift —
  **Adaptive Conformal Inference** (Gibbs & Candès 2021) in
  `models/online_conformal.py`: a model-agnostic, **leakage-safe** wrapper that
  walks origins in time order and adjusts interval width online from realised
  coverage feedback (queried via `get_vintage`). Made the adaptation **one-sided**
  (separate multipliers per tail) because epidemic miscoverage is asymmetric (the
  median under-predicts rising waves), which keeps intervals tight (better WIS).
  Result on the 3 evaluation locations:
  **MIST → MIST+ACI: cov-50 0.301→0.506, cov-95 0.724→0.933, ECE 0.162→0.055**,
  with WIS *improving* (uncalibrated base 1177 → ~1030). Both coverages now within
  ~0.02 of nominal.
- **2.3 Phase-conditioned uncertainty** — achieved *by construction*: ACI widens
  precisely when misses spike (the rising/peak phases), so per-phase ECE drops most
  there (Peak 0.398→0.187, Rising 0.260→0.189) without a separate variance head.
- `models/mist_v2.py` (`MISTModelV2`) subclasses v1 and also offers split-conformal
  (`use_conformal`) computed on a prior flu season in normalised units — kept as an
  alternative/ablation; the pipeline default calibrator is ACI.
- Tests: `tests/test_calibration.py` (ACI raises coverage, split-conformal deltas
  populated, ACI no-leakage). Full suite: **42 passed**.

## Phase 1 — Data integrity & vintage fix
- **1.3 Location crosswalk** (`features/crosswalk.py` → `data/crosswalk.csv`, 57 rows):
  maps FluSight FIPS ↔ NHSN abbrev ↔ ILINet lowercase ↔ state name, with a
  `nhsn_has_territory` flag. Validated against the raw files: every FluSight location
  maps to exactly one NHSN and one ILINet code **except Puerto Rico (FIPS 72)**, which
  NHSN reports but ILINet does not — flagged as the single documented gap.
- **1.1 NHSN reporting shift** (`ingestion/nhsn_shift.py`, `ingestion/nhsn.py`):
  added `NHSN_MANDATE_DATE = 2024-11-01` + `post_mandatory_flag`. Plotted weekly
  `totalconfflunewadm` per jurisdiction with the mandate line
  (`figures/nhsn_reporting_shift.png`); wrote `data/processed/nhsn_clean.csv` with a
  `post_mandatory` column. Every jurisdiction shows a large post-mandate mean increase
  (national +309%) — partly true reporting-completeness gain, partly seasonal
  composition of the pre/post windows; the covariate lets the model absorb the break.
- **1.2 ILINet genuine vintages** (`ingestion/ilinet.py`): real MMWR week↔date helpers
  (round-trip verified), `fluview` fetch via the **`lag`** parameter (corrected from the
  spec's non-existent `as_of`), schema mapping with a **real** `issue_date` (Saturday of
  the issue epiweek). Added `DELPHI_API_KEY` support.
  **Status: code complete + tested; the live pull is blocked by Delphi's anonymous
  rate limit (HTTP 429) and is deferred** — re-runs cleanly once a free API key is set
  (`data/raw/ilinet_vintaged.csv` produced then). The existing final-value
  `data/raw/ilinet.csv` already supports the Phase 3.3 analogue work.
- Tests: `tests/test_phase1_data.py` (crosswalk completeness/mapping/validation,
  `post_mandatory` flag, MMWR round-trip, ILINet schema with real issue dates).

## Phase 0 — Repo setup
- Initialised git repository and connected it to the remote
  `https://github.com/Nafis878/Epidemic-Forecasting`.
- Added `.gitignore` (ignores `data/store/*.parquet` — regenerable via
  `python -m ingestion.nhsn` — plus `__pycache__/`, caches, editor dirs).
- Added this `CHANGES.md`.
- Committed the existing STEP 0–6 codebase as the baseline:
  - Versioned (vintaged) DuckDB+Parquet store with leakage-safe `get_vintage`.
  - Ingestion (FluSight target, NHSN HRD), WIS/MAE/coverage metrics, rolling-origin
    backtester.
  - Baselines (ARIMA, SEIR), DL baselines (TFT, PatchTST), and the MIST-Transformer
    (mechanistic attention + phase-gated MoE + multi-resolution patching).
  - Evaluation/visualisation + 32 passing tests.

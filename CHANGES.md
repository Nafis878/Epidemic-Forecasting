# CHANGES

A running log of changes during the NeurIPS upgrade sprint. Newest first.

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

# CHANGES

A running log of changes during the NeurIPS upgrade sprint. Newest first.

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

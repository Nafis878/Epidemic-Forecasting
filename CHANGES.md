# CHANGES

A running log of changes during the NeurIPS upgrade sprint. Newest first.

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

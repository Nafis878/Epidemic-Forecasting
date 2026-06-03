# Data ingestion

All three data sources are downloaded into the leakage-safe `VersionedStore`
(`features/versioned_store.py`), which records the `issue_date` of every observation so
the backtester can ask "what was knowable as of date *t*?" Run the modules below from the
repository root.

## Sources

### 1. FluSight hospitalization target — `ingestion/nhsn.py`
- **What:** weekly confirmed influenza hospital admissions per location (the forecast
  target), plus the upstream NHSN Hospital Respiratory Data (HRD) feed.
- **FluSight truth:** `https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/`
  (CDC FluSight Forecast Hub target data). No key required.
- **NHSN HRD:** `https://data.cdc.gov/resource/ua7e-t2fy.csv`
  (fallback `https://data.cdc.gov/resource/mpgq-jmmr.csv`). Public Socrata endpoint.
- **Run:** `python -m ingestion.nhsn`

### 2. ILINet outpatient ILI% — `ingestion/ilinet.py`
- **What:** Delphi `fluview` weighted national + per-state ILI%, ~25-year history. Used as
  a leading indicator and for the historical-analogue (DTW) prior.
- **Endpoint:** `https://api.delphi.cmu.edu/epidata/fluview/`
- **API key:** set `DELPHI_API_KEY` in the environment. Register a free key at
  `https://api.delphi.cmu.edu/epidata/admin/registration_form`. Without a key the public
  endpoint is heavily rate-limited (HTTP 429); the loader backs off and pulls **genuine
  vintages** via the `lag` parameter, so a key is strongly recommended.
- **Run:** `DELPHI_API_KEY=... python -m ingestion.ilinet`

## Caveats

- **NHSN voluntary → mandatory reporting shift.** Reporting became mandatory on
  **2024-11-01** (`NHSN_MANDATE_DATE` in `ingestion/nhsn.py`). Pre- vs post-mandate
  volumes are not directly comparable, so downstream models receive a boolean
  `post_mandatory` covariate. `ingestion/nhsn_shift.py` characterises the shift and writes
  the processed NHSN frame with that column; `figures/nhsn_reporting_shift.png` visualises it.
- **Location crosswalk.** Three coding systems (FIPS, two-letter abbreviation, ILINet
  region codes) are unified through `data/crosswalk.csv` so FluSight, NHSN, and ILINet
  series align on a common 53-location panel.
- **Vintage approximation.** ILINet vintages are genuine (Delphi `lag`); NHSN/FluSight
  revisions are approximated where a true issue-dated archive is unavailable.

"""Ingest FluSight / NHSN influenza hospitalization data into the versioned store.

Two public sources are pulled and normalised into the store schema
``(source, signal, location, reference_date, issue_date, value)``:

* **FluSight ground truth** — ``target-hospital-admissions.csv`` from the
  CDC FluSight Forecast Hub. This is the *forecasting target* (weekly incident
  confirmed-flu hospital admissions). Locations are 2-digit FIPS codes
  (``US`` for national).
* **NHSN HRD** — the CDC Socrata dataset ``ua7e-t2fy`` (Weekly Hospital
  Respiratory Data by jurisdiction). The upstream raw source; locations are
  2-letter jurisdiction abbreviations (``USA`` national). The weekly
  confirmed-flu admission count lives in ``totalconfflunewadm``.

Reporting lag and ``issue_date``
--------------------------------
Neither source file carries true revision *vintages* (each is a single latest
snapshot). To make :meth:`VersionedStore.get_vintage` a meaningful anti-leakage
boundary we model **reporting lag**: a value describing the week ending
``reference_date`` only becomes *knowable* after the data are reported, so we set
``issue_date = reference_date + reporting_lag_days``. A forecaster standing on
``forecast_date`` therefore cannot see weeks that would not yet have been
reported. The lag is configurable (default 4 days: week ends Saturday, NHSN
reporting is due the following Tuesday).
"""

from __future__ import annotations

import io
import os
import sys
from typing import Optional

import pandas as pd
import requests

# Make the project root importable for `python -m ingestion.nhsn` and direct runs.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402

FLUSIGHT_URL = (
    "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/"
    "main/target-data/target-hospital-admissions.csv"
)
# Primary CDC Socrata resource + a mirror used as fallback.
NHSN_URL = "https://data.cdc.gov/resource/ua7e-t2fy.csv"
NHSN_URL_FALLBACK = "https://data.cdc.gov/resource/mpgq-jmmr.csv"

SIGNAL = "flu_hosp_admissions"
DEFAULT_REPORTING_LAG_DAYS = 4


# --------------------------------------------------------------------------- raw
def fetch_flusight_truth(timeout: int = 120) -> pd.DataFrame:
    """Download the FluSight target file -> [reference_date, location, value]."""
    r = requests.get(FLUSIGHT_URL, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    out = df.rename(columns={"date": "reference_date"})[
        ["reference_date", "location", "value"]
    ].copy()
    return out


def fetch_nhsn(timeout: int = 120, limit: int = 500_000) -> pd.DataFrame:
    """Download NHSN HRD flu admissions -> [reference_date, location, value]."""
    params = {"$limit": limit, "$select": "weekendingdate,jurisdiction,totalconfflunewadm"}
    try:
        r = requests.get(NHSN_URL, params=params, timeout=timeout)
        r.raise_for_status()
    except Exception:
        r = requests.get(NHSN_URL_FALLBACK, params=params, timeout=timeout)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    out = df.rename(
        columns={
            "weekendingdate": "reference_date",
            "jurisdiction": "location",
            "totalconfflunewadm": "value",
        }
    )[["reference_date", "location", "value"]].copy()
    return out


# ---------------------------------------------------------------------- normalise
def to_schema(
    df: pd.DataFrame,
    source: str,
    signal: str = SIGNAL,
    reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
) -> pd.DataFrame:
    """Map a [reference_date, location, value] frame to the store schema.

    ``issue_date`` is derived as ``reference_date + reporting_lag_days`` (see the
    module docstring), and rows with a missing value are dropped.
    """
    out = df.copy()
    out["reference_date"] = pd.to_datetime(out["reference_date"]).dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value", "reference_date", "location"])
    out["issue_date"] = out["reference_date"] + pd.Timedelta(days=reporting_lag_days)
    out["source"] = source
    out["signal"] = signal
    return out[SCHEMA_COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------------------ ingest
def ingest_flusight(store: VersionedStore, **kwargs) -> int:
    """Fetch FluSight truth and ingest it. Returns the number of rows stored."""
    schema_df = to_schema(fetch_flusight_truth(), source="flusight", **kwargs)
    store.ingest(schema_df)
    return len(schema_df)


def ingest_nhsn(store: VersionedStore, **kwargs) -> int:
    """Fetch NHSN HRD and ingest it. Returns the number of rows stored."""
    schema_df = to_schema(fetch_nhsn(), source="nhsn", **kwargs)
    store.ingest(schema_df)
    return len(schema_df)


def ingest_all(store: Optional[VersionedStore] = None,
               store_dir: str = "data/store") -> dict:
    """Ingest both sources into ``store`` (created at ``store_dir`` if not given)."""
    store = store or VersionedStore(store_dir=store_dir)
    counts = {
        "flusight": ingest_flusight(store),
        "nhsn": ingest_nhsn(store),
    }
    return counts


if __name__ == "__main__":
    counts = ingest_all()
    print("Ingested rows:", counts)
    store = VersionedStore()
    print("\nDistinct series stored (head):")
    print(store.list_signals().head(10).to_string(index=False))

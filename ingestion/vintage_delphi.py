"""Genuine issue-dated vintages from the Delphi Epidata covidcast API.

This replaces the *synthetic* vintage approximation in ``ingestion/nhsn.py``
(``issue_date = reference_date + fixed_lag``, which yields exactly one issue per
reference week — see ``evaluation/vintage_report.py``) with **real revision
history**. The Delphi covidcast endpoint returns, for each
``(geo_value, time_value)``, every dated revision as a separate row carrying its
own ``issue``; ingesting all of them makes ``get_vintage(..., as_of=t)`` exercise
genuine revision leakage, which is the core promise of the benchmark.

Signal catalogue (weekly hospital admissions; confirm against
https://cmu-delphi.github.io/delphi-epidata/api/covidcast_signals.html):

* Influenza  — ``nhsn`` / ``confirmed_admissions_influenza_ew`` (weekly), or the
  daily ``hhs`` / ``confirmed_admissions_influenza_1d`` aggregated to weeks.
* COVID-19   — ``nhsn`` / ``confirmed_admissions_covid_ew``  (weekly), or
  ``hhs`` / ``confirmed_admissions_covid_1d``.
* RSV        — ``nhsn`` / ``confirmed_admissions_rsv_ew`` (where available).

The network call is isolated in :func:`fetch_covidcast_issues`; the pure
transform :func:`normalise_covidcast` and the epiweek helper
:func:`epiweek_to_saturday` are unit-tested without the API. A free
``DELPHI_API_KEY`` (env var) lifts rate limits.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd
import requests

from features.versioned_store import SCHEMA_COLUMNS, VersionedStore

API_URL = "https://api.delphi.cmu.edu/epidata/covidcast/"

# Genuine-vintage weekly NHSN signals on Delphi covidcast (all carry real
# issue-dated revisions, and span 2020-2026 -> every benchmark season).
DELPHI_SOURCE = "nhsn"
DISEASE_SIGNALS = {
    "flu_hosp_admissions": "confirmed_admissions_flu_ew",
    "covid_hosp_admissions": "confirmed_admissions_covid_ew",
    "rsv_hosp_admissions": "confirmed_admissions_rsv_ew",
}

_HERE = os.path.dirname(__file__)
_CROSSWALK = os.path.abspath(os.path.join(_HERE, "..", "data", "crosswalk.csv"))


def load_geo_map(path: str = _CROSSWALK) -> dict:
    """covidcast geo_value (lowercase abbrev / 'us') -> store FIPS location."""
    cw = pd.read_csv(path, dtype=str)
    m = {row["abbrev_lower"]: row["fips"] for _, row in cw.iterrows()
         if isinstance(row["abbrev_lower"], str)}
    m["us"] = "US"                     # covidcast 'nation' geo_value is 'us'
    return m


# --------------------------------------------------------------- epiweek helper
def epiweek_to_saturday(epiweek: int) -> date:
    """Map an MMWR epiweek (``YYYYWW``) to its week-ending **Saturday** date.

    MMWR week 1 is the Sun--Sat week containing Jan 4; week ``W`` ends 7*(W-1)
    days after week 1's Saturday. E.g. ``202401 -> 2024-01-06``,
    ``202340 -> 2023-10-07``.
    """
    year, week = divmod(int(epiweek), 100)
    jan4 = date(year, 1, 4)
    sun_index = (jan4.weekday() + 1) % 7           # Mon=0..Sun=6  ->  Sun=0..Sat=6
    week1_sat = jan4 + timedelta(days=6 - sun_index)
    return week1_sat + timedelta(days=7 * (week - 1))


def _to_reference_date(time_value: int, time_type: str) -> pd.Timestamp:
    if time_type == "week":
        return pd.Timestamp(epiweek_to_saturday(int(time_value)))
    return pd.Timestamp(pd.to_datetime(str(time_value), format="%Y%m%d"))


# --------------------------------------------------------------- pure transform
def normalise_covidcast(rows: Iterable[dict], *, signal: str, source: str,
                        time_type: str = "week", geo_map: Optional[dict] = None) -> pd.DataFrame:
    """Map covidcast ``epidata`` rows to the store schema (one row per revision).

    Each input row must carry ``geo_value``, ``time_value``, ``issue`` (both
    epiweeks/dates per ``time_type``) and ``value``. ``issue`` becomes a genuine
    ``issue_date``, so multiple revisions of the same reference week are kept.
    If ``geo_map`` is given, ``geo_value`` is mapped through it (e.g. covidcast
    lowercase abbrev -> store FIPS); rows whose geo is unmapped are skipped.
    """
    recs = []
    for r in rows:
        if r.get("value") is None:
            continue
        gv = str(r["geo_value"]).lower()
        if geo_map is not None:
            loc = geo_map.get(gv)
            if loc is None:
                continue
        else:
            loc = gv.upper()
        recs.append({
            "source": source,
            "signal": signal,
            "location": loc,
            "reference_date": _to_reference_date(r["time_value"], time_type),
            "issue_date": _to_reference_date(r["issue"], time_type),
            "value": float(r["value"]),
        })
    df = pd.DataFrame.from_records(recs, columns=SCHEMA_COLUMNS)
    if not df.empty:
        df = df.sort_values(["location", "reference_date", "issue_date"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------------- network
def fetch_covidcast_issues(data_source: str, api_signal: str, *, geo_type: str,
                           time_values: str, geo_value: str = "*",
                           time_type: str = "week", issues: Optional[str] = None,
                           api_key: Optional[str] = None, timeout: int = 120,
                           max_retries: int = 6) -> list[dict]:
    """Fetch dated revisions (every ``issue``) for a covidcast signal.

    ``time_values`` selects the reference periods (a covidcast range string,
    e.g. ``"202240-202320"``); ``issues`` selects which revisions to return
    (defaults to ``time_values``). The keyless endpoint caps the response and
    occasionally returns a non-JSON rate-limit page, so we keep per-call result
    sets small (see :func:`ingest_panel` chunking) and back off on 429 / non-JSON.
    """
    params = {
        "data_source": data_source, "signal": api_signal, "time_type": time_type,
        "geo_type": geo_type, "time_values": time_values, "geo_value": geo_value,
        "issues": issues or time_values,
    }
    key = api_key or os.environ.get("DELPHI_API_KEY")
    if key:
        params["api_key"] = key
    backoff = 2.0
    for attempt in range(max_retries):
        resp = requests.get(API_URL, params=params, timeout=timeout)
        if resp.status_code == 429:                 # rate limited -> back off
            time.sleep(backoff); backoff *= 2
            continue
        if resp.status_code != 200:
            resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:                          # non-JSON (rate-limit page) -> retry
            time.sleep(backoff); backoff *= 2
            continue
        if payload.get("result") == 1:
            return payload.get("epidata", [])
        if payload.get("result") == -2:             # no data for this slice
            return []
        # result == -1 etc. (often "too many rows") -> let the caller chunk smaller
        raise RuntimeError(f"Epidata result={payload.get('result')}: {payload.get('message')}")
    raise RuntimeError("Exhausted retries fetching covidcast issues")


def _ew_halfyear_chunks(start_ew: int, end_ew: int) -> list[str]:
    """Half-year epiweek ranges 'YYYYWW-YYYYWW' spanning [start_ew, end_ew].

    Keeps each per-geo query small (≈26 weeks x revisions) so the keyless API
    row cap is not hit even for old, heavily-backfilled weeks.
    """
    sy, ey = start_ew // 100, end_ew // 100
    out = []
    for y in range(sy, ey + 1):
        for lo_w, hi_w in ((1, 26), (27, 53)):
            lo, hi = y * 100 + lo_w, y * 100 + hi_w
            lo, hi = max(lo, start_ew), min(hi, end_ew)
            if lo <= hi:
                out.append(f"{lo}-{hi}")
    return out


def ingest_genuine(store: VersionedStore, data_source: str, api_signal: str, *,
                   signal: str, geo_type: str, time_values: str,
                   time_type: str = "week", source: Optional[str] = None,
                   **fetch_kwargs) -> int:
    """Fetch genuine vintages for one signal and ingest them. Returns rows stored."""
    source = source or f"delphi_{data_source}"
    rows = fetch_covidcast_issues(data_source, api_signal, geo_type=geo_type,
                                  time_values=time_values, time_type=time_type,
                                  **fetch_kwargs)
    df = normalise_covidcast(rows, signal=signal, source=source, time_type=time_type)
    if df.empty:
        return 0
    store.ingest(df)
    return len(df)


def ingest_panel(store: VersionedStore, *, api_signal: str, signal: str,
                 time_values: str, geo_map: dict, states_lower: list[str],
                 source: str = DELPHI_SOURCE, sleep: float = 0.4,
                 verbose: bool = True) -> int:
    """Ingest one disease's genuine weekly vintages for the whole panel.

    Pulls every dated revision per geo (states one-by-one + the national
    aggregate, since the keyless public endpoint rejects all-geo + all-issue
    bulk queries), maps geos to FIPS, and ingests once. Returns rows stored.
    """
    start_ew, end_ew = (int(x) for x in time_values.split("-"))
    chunks = _ew_halfyear_chunks(start_ew, end_ew)
    frames = []
    failed = 0
    targets = [("state", g) for g in states_lower] + [("nation", "us")]
    for geo_type, geo_value in targets:
        geo_rows = 0
        for chunk in chunks:
            clo = chunk.split("-")[0]
            try:                                    # all revisions of the chunk's weeks
                rows = fetch_covidcast_issues(source, api_signal, geo_type=geo_type,
                                              time_values=chunk, geo_value=geo_value,
                                              issues=f"{clo}-{end_ew}", time_type="week")
            except Exception as e:                  # skip a flaky chunk, keep going
                failed += 1
                if verbose:
                    print(f"    [skip {geo_value} {chunk}] {type(e).__name__}: {str(e)[:70]}")
                time.sleep(sleep)
                continue
            df = normalise_covidcast(rows, signal=signal, source=source,
                                     time_type="week", geo_map=geo_map)
            if not df.empty:
                frames.append(df)
                geo_rows += len(df)
            time.sleep(sleep)
        if verbose:
            print(f"    {geo_value:4s} {api_signal}: {geo_rows} revision rows")
    if failed:
        print(f"  WARNING: {failed} chunk(s) dropped after retries (rate limiting). "
              f"Set DELPHI_API_KEY to avoid throttling, then re-run — ingestion is "
              f"idempotent (duplicate revisions are de-duped).")
    if not frames:
        return 0
    alldf = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["source", "signal", "location", "reference_date", "issue_date"])
    store.ingest(alldf)
    return len(alldf)


def ingest_genuine_panel(store_dir: str = "data/store_genuine",
                         diseases: Optional[list[str]] = None,
                         time_values: str = "202201-202526",
                         crosswalk: str = _CROSSWALK) -> dict:
    """Build the genuine-vintage multi-disease store (flu/covid/rsv) from NHSN."""
    diseases = diseases or list(DISEASE_SIGNALS)
    geo_map = load_geo_map(crosswalk)
    cw = pd.read_csv(crosswalk, dtype=str)
    states_lower = [a for a, f in zip(cw["abbrev_lower"], cw["fips"])
                    if isinstance(a, str) and f != "US"]
    store = VersionedStore(store_dir=store_dir)
    counts = {}
    for disease in diseases:
        print(f"=== {disease} ({DISEASE_SIGNALS[disease]}) ===")
        counts[disease] = ingest_panel(
            store, api_signal=DISEASE_SIGNALS[disease], signal=disease,
            time_values=time_values, geo_map=geo_map, states_lower=states_lower)
        print(f"  -> {counts[disease]} rows")
    return counts


if __name__ == "__main__":  # pragma: no cover - live API (needs network)
    import argparse

    ap = argparse.ArgumentParser(description="Ingest genuine multi-disease Delphi vintages.")
    ap.add_argument("--store-dir", default="data/store_genuine")
    ap.add_argument("--diseases", nargs="*", default=None)
    ap.add_argument("--time-values", default="202201-202526")
    args = ap.parse_args()
    counts = ingest_genuine_panel(store_dir=args.store_dir, diseases=args.diseases,
                                  time_values=args.time_values)
    print("ingested:", counts)

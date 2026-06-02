"""Ingest ILINet (Delphi ``fluview``) outpatient ILI%% with **genuine vintages**.

ILINet reports the weighted/unweighted percent of outpatient visits for
influenza-like illness. Unlike the FluSight/NHSN snapshots (which we lag
synthetically), ILINet revisions are *real* and retrievable: the ``fluview``
endpoint exposes historical issues via the ``issues``/``lag`` parameters (it has
**no** ``as_of`` parameter). We pull, for a window of issue-epiweeks, the data as
it actually stood at each issue, so the store's ``issue_date`` is genuine rather
than ``reference_date + 4 days``.

Store mapping
-------------
* ``source="ilinet"``, ``signal="ili_pct"``
* ``location`` = ILINet region code (``nat`` national, ``ca`` California, ...)
* ``reference_date`` = Saturday ending the MMWR reference epiweek (FluSight cadence)
* ``issue_date``     = Saturday ending the MMWR **issue** epiweek (real vintage)
* ``value``          = ``wili`` for ``nat`` (weighted), else ``ili`` (unweighted)

Run ``python -m ingestion.ilinet`` to fetch vintages over the modelling window,
write ``data/raw/ilinet_vintaged.csv`` and report how many values differ from the
final (synthetic-lag) version.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402

FLUVIEW_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
SIGNAL = "ili_pct"
SOURCE = "ilinet"
RAW_VINTAGED_PATH = "data/raw/ilinet_vintaged.csv"

# All ILINet state/territory codes plus the national series, as present in the
# existing raw pull (no Puerto Rico — see features/crosswalk.py).
ALL_REGIONS = [
    "nat", "ak", "al", "ar", "az", "ca", "co", "ct", "dc", "de", "fl", "ga",
    "hi", "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi",
    "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv", "ny",
    "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt",
    "wa", "wi", "wv", "wy",
]


# --------------------------------------------------------------- MMWR week <-> date
def _mmwr_week1_start(year: int) -> date:
    """Sunday starting MMWR week 1 of ``year`` (week with >=4 days in January)."""
    jan1 = date(year, 1, 1)
    wday = (jan1.weekday() + 1) % 7        # Sunday=0 .. Saturday=6
    if wday <= 3:                          # Jan 1 is Sun/Mon/Tue/Wed -> its week is week 1
        return jan1 - timedelta(days=wday)
    return jan1 + timedelta(days=(7 - wday))  # else week 1 starts the next Sunday


def epiweek_to_saturday(epiweek: int) -> pd.Timestamp:
    """MMWR ``YYYYWW`` -> Timestamp of the Saturday ending that week."""
    year, week = divmod(int(epiweek), 100)
    sunday = _mmwr_week1_start(year) + timedelta(weeks=week - 1)
    return pd.Timestamp(sunday + timedelta(days=6))


def date_to_epiweek(d) -> int:
    """Timestamp/date -> MMWR ``YYYYWW`` of the week containing ``d``."""
    d = pd.Timestamp(d).date()
    # Sunday starting d's week.
    wday = (d.weekday() + 1) % 7
    week_sunday = d - timedelta(days=wday)
    # The MMWR year is the year of the week's Wednesday (Sunday + 3).
    wednesday = week_sunday + timedelta(days=3)
    year = wednesday.year
    w1 = _mmwr_week1_start(year)
    week = (week_sunday - w1).days // 7 + 1
    return year * 100 + week


def epiweek_range(start_ew: int, end_ew: int) -> list[int]:
    """Inclusive list of valid MMWR epiweeks from ``start_ew`` to ``end_ew``."""
    out, d = [], epiweek_to_saturday(start_ew)
    end = epiweek_to_saturday(end_ew)
    while d <= end:
        out.append(date_to_epiweek(d))
        d += pd.Timedelta(days=7)
    return out


# -------------------------------------------------------------------------- fetch
def _fluview(regions, epiweeks: str, issues: Optional[str] = None,
             lag: Optional[int] = None, timeout: int = 120,
             max_retries: int = 10) -> pd.DataFrame:
    """One fluview call with backoff on rate-limiting (HTTP 429).

    ``epiweeks``/``issues`` are API range strings; ``lag`` (weeks between epiweek
    and issue) is an alternative vintage selector (mutually exclusive with
    ``issues``).
    """
    params = {"regions": ",".join(regions), "epiweeks": epiweeks}
    if issues is not None:
        params["issues"] = issues
    if lag is not None:
        params["lag"] = lag
    # A free Delphi API key lifts the strict anonymous rate limit. Register at
    # https://api.delphi.cmu.edu/epidata/admin/registration_form and export it as
    # DELPHI_API_KEY; without one the public endpoint is heavily throttled (429).
    api_key = os.environ.get("DELPHI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    for attempt in range(max_retries):
        r = requests.get(FLUVIEW_URL, params=params, timeout=timeout)
        if r.status_code == 429:                       # rate limited -> wait & retry
            wait = min(120.0, 10.0 * (2 ** attempt))   # cap per-sleep at 120s
            print(f"    [429] rate limited; sleeping {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        js = r.json()
        if js.get("result") != 1:
            # result == -2 means "no data"; anything else is an error worth surfacing.
            if js.get("result") == -2:
                return pd.DataFrame()
            raise RuntimeError(
                f"fluview error: result={js.get('result')} {js.get('message')}")
        return pd.DataFrame(js["epidata"])
    raise RuntimeError("fluview: exceeded retries due to repeated HTTP 429 rate limiting")


def fetch_vintaged(ref_start_ew: int, ref_end_ew: int, max_lag: int = 4,
                   regions: Iterable[str] = ALL_REGIONS, region_chunk: int = 60,
                   verbose: bool = True) -> pd.DataFrame:
    """Genuine vintages via the ``lag`` parameter (few calls -> avoids rate limits).

    For each ``lag`` in ``0..max_lag`` we request every reference epiweek in
    ``[ref_start_ew, ref_end_ew]`` *as issued ``lag`` weeks after it*. ``lag=0`` is
    the first-reported (real-time) value; higher lags are successive revisions.
    The API returns the genuine ``issue`` epiweek per row, which we use directly.
    This needs only ``(max_lag + 1) * ceil(len(regions)/region_chunk)`` calls.
    """
    regions = list(regions)
    eweeks = f"{ref_start_ew}-{ref_end_ew}"
    frames = []
    for lag in range(max_lag + 1):
        for i in range(0, len(regions), region_chunk):
            grp = regions[i:i + region_chunk]
            df = _fluview(grp, epiweeks=eweeks, issues=None, lag=lag)
            if not df.empty:
                frames.append(df)
            if verbose:
                print(f"  lag={lag} regions {grp[0]}..{grp[-1]}: "
                      f"{0 if df.empty else len(df)} rows")
            time.sleep(2.0)  # be polite to the public API (avoid 429 rate limiting)
    if not frames:
        return pd.DataFrame()
    # Drop exact duplicate (location, reference, issue) rows across lag calls.
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["region", "epiweek", "issue"])


def fetch_final(regions: Iterable[str] = ALL_REGIONS, epiweeks: str = "199740-202539",
                verbose: bool = True) -> pd.DataFrame:
    """The latest (final) value per epiweek over a long range (for analogues)."""
    frames = []
    regions = list(regions)
    for i in range(0, len(regions), 10):
        grp = regions[i:i + 10]
        df = _fluview(grp, epiweeks=epiweeks)
        if not df.empty:
            frames.append(df)
        if verbose:
            print(f"  final {grp[0]}..{grp[-1]}: {0 if df.empty else len(df)} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------- normalise
def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """fluview rows -> store schema with genuine ``issue_date``."""
    out = df.copy()
    # value: weighted ILI for national, unweighted for states.
    val = out["ili"].astype(float)
    if "wili" in out.columns:
        val = val.where(out["region"] != "nat", out["wili"].astype(float))
    out["value"] = val
    out["reference_date"] = out["epiweek"].map(epiweek_to_saturday)
    out["issue_date"] = out["issue"].map(epiweek_to_saturday)
    out["location"] = out["region"].astype(str)
    out["source"] = SOURCE
    out["signal"] = SIGNAL
    out = out.dropna(subset=["value", "reference_date", "issue_date"])
    return out[SCHEMA_COLUMNS].reset_index(drop=True)


def revision_report(vintaged: pd.DataFrame, final_path: str = "data/raw/ilinet.csv") -> None:
    """Print how many first-reported values differ from the final (synthetic) value."""
    try:
        fin = pd.read_csv(final_path)
    except FileNotFoundError:
        print("  (no final ilinet.csv to compare against)")
        return
    fin_val = fin["ili"].astype(float)
    if "wili" in fin.columns:
        fin_val = fin_val.where(fin["region"] != "nat", fin["wili"].astype(float))
    fin_map = {(r, int(e)): v for r, e, v in zip(fin["region"], fin["epiweek"], fin_val)}

    v = vintaged.copy()
    v["epiweek"] = v["reference_date"].map(date_to_epiweek)
    first = (v.sort_values("issue_date")
             .groupby(["location", "epiweek"], as_index=False).first())
    diff = same = miss = 0
    for loc, ew, val in zip(first["location"], first["epiweek"], first["value"]):
        f = fin_map.get((loc, int(ew)))
        if f is None:
            miss += 1
        elif abs(f - val) > 1e-9:
            diff += 1
        else:
            same += 1
    tot = diff + same
    pct = 100.0 * diff / tot if tot else 0.0
    print(f"\nRevision sanity check (first-reported vs final value):")
    print(f"  {diff}/{tot} first-reported values differ from final ({pct:.1f}%); "
          f"{same} unchanged; {miss} not in final file.")


# ------------------------------------------------------------------------- ingest
def ingest_vintaged(store: Optional[VersionedStore] = None,
                    ref_start_ew: int = 202201, ref_end_ew: int = 202418,
                    max_lag: int = 2, store_dir: str = "data/store") -> int:
    """Fetch genuine ILINet vintages, save CSV, and ingest into the store."""
    vint = fetch_vintaged(ref_start_ew, ref_end_ew, max_lag=max_lag)
    if vint.empty:
        raise RuntimeError("fluview returned no vintaged rows (network/endpoint issue?)")
    schema_df = to_schema(vint)
    os.makedirs(os.path.dirname(RAW_VINTAGED_PATH) or ".", exist_ok=True)
    schema_df.to_csv(RAW_VINTAGED_PATH, index=False)
    print(f"Wrote {RAW_VINTAGED_PATH} ({len(schema_df)} vintaged rows, "
          f"{schema_df['location'].nunique()} locations).")
    revision_report(schema_df)

    store = store or VersionedStore(store_dir=store_dir)
    store.ingest(schema_df)
    return len(schema_df)


if __name__ == "__main__":
    n = ingest_vintaged()
    print(f"\nIngested {n} ILINet vintaged rows into the store.")

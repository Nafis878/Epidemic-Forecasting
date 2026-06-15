"""Vintage-authenticity report for the versioned store.

A benchmark whose central claim is *leakage-safe vintage evaluation* must be able
to prove its vintages are **genuine** — i.e. that a value for a given
``reference_date`` actually carries multiple dated revisions (``issue_date``), so
that ``get_vintage(..., as_of=t)`` returns something materially different from the
finally-revised series. A pipeline that fabricates vintages as
``issue_date = reference_date + fixed_lag`` has exactly **one** issue per
reference week and therefore offers *no* protection against revision leakage —
only against not-yet-reported leakage.

This module audits the store and, per ``(source, signal)``, reports:

* ``mean_issues_per_ref`` — average number of distinct ``issue_date`` revisions per
  (location, reference_date). ``1.0`` means synthetic / single-snapshot vintages.
* ``pct_revised`` — fraction of (location, reference_date) cells that were revised
  at least once.
* ``mean_rel_revision`` — mean relative gap ``(max-min)/max`` across a cell's
  revisions (revision magnitude), ``0`` when never revised.
* ``genuine_vintage`` — ``True`` iff revisions actually exist.

Output: ``results/vintage_authenticity.csv`` (and a printed table). This is the
honest diagnostic the Datasets & Benchmarks paper must include; it is also the
acceptance check for the genuine-vintage ingestion work (WP1).
"""

from __future__ import annotations

import argparse
import os

import duckdb
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
DEFAULT_STORE = os.path.join(ROOT, "data", "store")


def _glob(store_dir: str) -> str:
    return os.path.join(store_dir, "*.parquet").replace("\\", "/")


def audit(store_dir: str = DEFAULT_STORE) -> pd.DataFrame:
    """Return a per-(source, signal) vintage-authenticity table."""
    g = _glob(store_dir)
    con = duckdb.connect(":memory:")
    q = f"""
    WITH cells AS (
        SELECT source, signal, location, reference_date,
               COUNT(DISTINCT issue_date)                       AS n_issues,
               MAX(value)                                       AS vmax,
               MIN(value)                                       AS vmin
        FROM read_parquet('{g}')
        GROUP BY source, signal, location, reference_date
    )
    SELECT source, signal,
           COUNT(*)                                             AS cells,
           COUNT(DISTINCT location)                             AS locations,
           AVG(n_issues)                                        AS mean_issues_per_ref,
           MAX(n_issues)                                        AS max_issues_per_ref,
           AVG(CASE WHEN n_issues > 1 THEN 1.0 ELSE 0.0 END)    AS pct_revised,
           AVG(CASE WHEN vmax > 0 THEN (vmax - vmin) / vmax ELSE 0.0 END) AS mean_rel_revision
    FROM cells
    GROUP BY source, signal
    ORDER BY source, signal
    """
    df = con.execute(q).fetchdf()
    df["genuine_vintage"] = df["mean_issues_per_ref"] > 1.0 + 1e-9
    return df


def run(store_dir: str = DEFAULT_STORE, out_path: str | None = None) -> pd.DataFrame:
    df = audit(store_dir)
    out_path = out_path or os.path.join(RES, "vintage_authenticity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("=== vintage authenticity ===")
    print(df.round(3).to_string(index=False))
    n_genuine = int(df["genuine_vintage"].sum())
    print(f"\n{n_genuine}/{len(df)} (source, signal) streams have GENUINE vintages.")
    if n_genuine < len(df):
        print("WARNING: streams with mean_issues_per_ref == 1.0 use synthetic "
              "(reporting-lag) vintages only; revision leakage is NOT exercised.")
    print(f"saved -> {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(store_dir=args.store, out_path=args.out)

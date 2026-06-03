"""Pooled 3-season Diebold-Mariano test (rising phase), mist_v2 vs all.

The single-season (2023-24) headline DM test puts ``mist_v2`` vs ARIMA in the rising
phase at p~=0.056 — suggestive but not significant. Pooling the per-forecast loss
differentials across all three seasons (2022-23, 2023-24, 2024-25) increases power.
This script reports the exact pooled p-value either way; it does **not** hide a result
that stays above 0.05.

Consumes ``results/season_records.csv`` (written by ``run_seasons``), labels each row
with its epidemic phase via :func:`evaluation.visualizer.attach_phase`, and writes
``tables/dm_headline_rising_3seasons.csv``.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dm_test import dm_test, headline_report  # noqa: E402
from evaluation.visualizer import attach_phase  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES, TAB = os.path.join(ROOT, "results"), os.path.join(ROOT, "tables")
SIG, SRC = "flu_hosp_admissions", "flusight"


def _load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RES, "season_records.csv"),
                     parse_dates=["forecast_date", "reference_date"],
                     dtype={"location": str})
    store = VersionedStore()
    panel = df["location"].astype(str).unique().tolist()
    truth = pd.concat([store.get_vintage(SIG, l, "2100-01-01", source=SRC) for l in panel],
                      ignore_index=True)
    return attach_phase(df, truth)


def run() -> pd.DataFrame:
    os.makedirs(TAB, exist_ok=True)
    df = _load()
    rising = df[df["phase"] == "Rising"]
    seasons = sorted(df["season"].unique())
    print(f"pooled seasons: {seasons} | rising-phase rows: {len(rising)}")

    head = headline_report(rising, focal="mist_v2", metric="wis", phase=None)
    head.to_csv(os.path.join(TAB, "dm_headline_rising_3seasons.csv"), index=False)
    print("\n=== mist_v2 vs all, RISING phase, 3 seasons pooled (mean WIS diff; <0 => MIST better) ===")
    print(head.round(4).to_string(index=False))

    if "arima" in rising["model"].unique():
        r = dm_test(rising[rising["model"] == "mist_v2"],
                    rising[rising["model"] == "arima"], metric="wis")
        verdict = "SIGNIFICANT (p<0.05)" if r["p"] < 0.05 else "NOT significant (p>=0.05)"
        print(f"\nmist_v2 vs arima (rising, pooled): mean_diff={r['mean_diff']:.3f} "
              f"p={r['p']:.4f} n={r['n']}  ->  {verdict}")
    print(f"\nsaved -> {os.path.join(TAB, 'dm_headline_rising_3seasons.csv')}")
    return head


if __name__ == "__main__":
    run()

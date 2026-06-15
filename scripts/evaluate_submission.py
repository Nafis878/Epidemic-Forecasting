"""Score an external submission against the benchmark (forecast-hub style).

A benchmark needs a documented, single-entry way for others to place a model on
the leaderboard. Following the FluSight / COVID-19 Forecast Hub model, a
contributor submits **forecasts** (not code): a long-format quantile file with
columns

    [forecast_date, location, horizon, reference_date, quantile, value]
    (an optional `season` column is inferred from forecast_date if absent)

produced from the benchmark's leakage-safe vintage inputs (use
`store.get_vintage(signal, location, forecast_date)` as the only data access).
This script joins the **final** observed truth from the versioned store, scores
every forecast with the weighted interval score, and reports per-season and
season-unweighted WIS plus 50/95% coverage — directly comparable to
`results/season_leaderboard.csv` / `results/multiseason_summary.csv`.

    python scripts/evaluate_submission.py --forecasts my_model.parquet --name my_model

The benchmark's own base models can be re-scored as a sanity check by passing a
single-model slice of `results/quantiles_long.parquet`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import _FINAL_AS_OF  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    interval_coverage, weighted_interval_score,
)
from evaluation.run_seasons import SEASONS, SIG, SRC  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402

REQUIRED = ["forecast_date", "location", "horizon", "reference_date", "quantile", "value"]


def _infer_season(forecast_date: pd.Timestamp) -> str:
    for name, cfg in SEASONS.items():
        lo, hi = pd.Timestamp(cfg["origins"][0]), pd.Timestamp(cfg["origins"][1])
        if lo <= forecast_date <= hi:
            return name
    return "out_of_season"


def _truth_map(store: VersionedStore, location: str) -> dict:
    t = store.get_vintage(SIG, location, _FINAL_AS_OF, source=SRC)
    return {pd.Timestamp(d): float(v) for d, v in zip(t["reference_date"], t["value"])}


def score(forecasts: pd.DataFrame, store: VersionedStore | None = None) -> pd.DataFrame:
    """Per-forecast WIS / coverage for a submission, truth from the vintage store."""
    missing = [c for c in REQUIRED if c not in forecasts.columns]
    if missing:
        raise ValueError(f"submission missing columns {missing}; need {REQUIRED}")
    store = store or VersionedStore()
    df = forecasts.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["location"] = df["location"].astype(str)
    if "season" not in df.columns:
        df["season"] = df["forecast_date"].map(_infer_season)

    truth_by_loc = {loc: _truth_map(store, loc) for loc in df["location"].unique()}

    rows = []
    keys = ["season", "forecast_date", "location", "horizon", "reference_date"]
    for key, g in df.groupby(keys):
        g = g.sort_values("quantile")
        y = truth_by_loc.get(key[2], {}).get(pd.Timestamp(key[4]))
        if y is None or np.isnan(y):
            continue
        qv = g["value"].to_numpy(float)
        ql = g["quantile"].to_numpy(float)
        rows.append({**dict(zip(keys, key)), "y_true": y,
                     "wis": float(weighted_interval_score(y, qv, ql)),
                     "cov_50": float(interval_coverage(y, qv, 0.50, ql)),
                     "cov_95": float(interval_coverage(y, qv, 0.95, ql))})
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame, name: str) -> pd.DataFrame:
    per_season = scored.groupby("season")["wis"].mean()
    out = {
        "model": name,
        "n_forecasts": int(len(scored)),
        "wis_unweighted": float(per_season.mean()),
        "wis_weighted": float(scored["wis"].mean()),
        "cov_50": float(scored["cov_50"].mean()),
        "cov_95": float(scored["cov_95"].mean()),
    }
    for s, v in per_season.items():
        out[f"wis_{s}"] = float(v)
    return pd.DataFrame([out])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", required=True, help="long-format quantile parquet/csv")
    ap.add_argument("--name", default="submission")
    ap.add_argument("--model-col", default=None,
                    help="if the file holds many models, the column + value to select, "
                         "e.g. 'model=patchtst'")
    args = ap.parse_args()

    path = args.forecasts
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if args.model_col and "=" in args.model_col:
        col, val = args.model_col.split("=", 1)
        df = df[df[col].astype(str) == val]
    scored = score(df)
    if scored.empty:
        print("No scorable forecasts (no realised targets matched). Check dates/locations.")
        return 1
    summary = summarize(scored, args.name)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

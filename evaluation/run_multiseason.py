"""Multi-season-first aggregation: the unified leaderboard + stratifications.

This is the benchmark the upgraded paper is built around. It assembles, into one
long frame, every base model (from ``results/quantiles_long.parquet``), the three
ensemble baselines (:mod:`evaluation.ensembles`), and the three hybrid ablations
(:mod:`models.phase_stack`), then reports — across all three seasons:

* ``results/season_leaderboard.csv``  — per (model, season) mean WIS / MAE /
  50% & 95% coverage.
* ``results/multiseason_summary.csv``  — per model the **season-unweighted** mean
  WIS (each season equal — the headline, so a single big season can't dominate)
  and the **forecast-weighted** (pooled) mean WIS, plus pooled coverage and a
  rank.
* ``results/phase_by_season.csv``      — per (model, season, vintage phase) WIS.
* ``results/horizon_by_season.csv``    — per (model, season, horizon) WIS.

Phase here is ``phase_origin`` — the vintage epidemic phase at the forecast
origin (leakage-safe, the same signal the hybrid gates on), not a final-truth
label of the target week.

Numbers written under the ``quick`` dump are pipeline-validation only; paper
numbers come exclusively from a ``--full`` dump (see ``scripts/reproduce.py``).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.ensembles import (  # noqa: E402
    LONG_PATH, build_ensembles, load_long, per_forecast_metrics,
)
from models.phase_stack import build_all_stacks  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))


def assemble_all(path: str = LONG_PATH) -> pd.DataFrame:
    """Base models + ensembles + hybrid ablations as one long frame."""
    long = load_long(path)
    ens = build_ensembles(long)
    stacks = build_all_stacks(long)
    return pd.concat([long, ens, stacks], ignore_index=True)


def _season_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    g = (metrics.groupby(["model", "season"])
         .agg(n=("wis", "size"), wis=("wis", "mean"), mae=("mae", "mean"),
              cov_50=("cov_50", "mean"), cov_95=("cov_95", "mean"))
         .reset_index())
    return g.sort_values(["season", "wis"]).reset_index(drop=True)


def _multiseason_summary(leaderboard: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    # Season-unweighted: mean of per-season mean WIS (each season equal weight).
    unw = leaderboard.groupby("model")["wis"].mean().rename("wis_unweighted")
    # Forecast-weighted (pooled) across all seasons.
    pooled = (metrics.groupby("model")
              .agg(wis_weighted=("wis", "mean"), mae=("mae", "mean"),
                   cov_50=("cov_50", "mean"), cov_95=("cov_95", "mean"),
                   n=("wis", "size")))
    out = pooled.join(unw).reset_index()
    out = out.sort_values("wis_unweighted").reset_index(drop=True)
    out.insert(1, "rank_unweighted", range(1, len(out) + 1))
    cols = ["model", "rank_unweighted", "wis_unweighted", "wis_weighted",
            "mae", "cov_50", "cov_95", "n"]
    return out[cols]


def _strat(metrics: pd.DataFrame, by: str) -> pd.DataFrame:
    g = (metrics.groupby(["model", "season", by])
         .agg(n=("wis", "size"), wis=("wis", "mean")).reset_index())
    return g.sort_values(["season", by, "wis"]).reset_index(drop=True)


def run(path: str = LONG_PATH) -> dict:
    os.makedirs(RES, exist_ok=True)
    allf = assemble_all(path)
    metrics = per_forecast_metrics(allf)

    leaderboard = _season_leaderboard(metrics)
    summary = _multiseason_summary(leaderboard, metrics)
    phase = _strat(metrics, "phase_origin")
    horizon = _strat(metrics, "horizon")

    leaderboard.to_csv(os.path.join(RES, "season_leaderboard.csv"), index=False)
    summary.to_csv(os.path.join(RES, "multiseason_summary.csv"), index=False)
    phase.to_csv(os.path.join(RES, "phase_by_season.csv"), index=False)
    horizon.to_csv(os.path.join(RES, "horizon_by_season.csv"), index=False)

    print("=== multiseason summary (season-unweighted WIS; lower=better) ===")
    print(summary.round(3).to_string(index=False))
    return {"leaderboard": leaderboard, "summary": summary,
            "phase": phase, "horizon": horizon}


if __name__ == "__main__":
    run()

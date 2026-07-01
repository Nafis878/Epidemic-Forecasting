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
* ``results/multiseason_common_summary.csv`` — the paper-facing paired version
  restricted to forecast keys present for every model, plus
  ``results/common_mask_report.csv`` explaining retained/dropped keys.
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
_FAIR_KEY = ["season", "forecast_date", "location", "horizon"]


def fair_key_cols(df: pd.DataFrame) -> list[str]:
    return (["disease"] if "disease" in df.columns else []) + _FAIR_KEY


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


def common_mask_metrics(metrics: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Return metrics restricted to forecast keys present for every selected model.

    The default headline leaderboard should be a paired comparison: every model
    receives exactly the same ``(season, forecast_date, location, horizon)`` cells.
    Available-case means are still useful diagnostics, but a common mask removes
    ambiguity when one model drops early/stale targets.
    """
    models = sorted(models or metrics["model"].unique())
    sub = metrics[metrics["model"].isin(models)].copy()
    fkey = fair_key_cols(sub)
    present = sub.drop_duplicates(fkey + ["model"])
    counts = present.groupby(fkey)["model"].nunique()
    complete = counts[counts == len(models)].index
    if len(complete) == 0:
        return sub.iloc[0:0].copy()
    keep = pd.MultiIndex.from_frame(sub[fkey]).isin(complete)
    return sub[keep].reset_index(drop=True)


def common_mask_report(metrics: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Per-season accounting for the common-mask comparison."""
    models = sorted(models or metrics["model"].unique())
    sub = metrics[metrics["model"].isin(models)]
    fkey = fair_key_cols(sub)
    present = sub.drop_duplicates(fkey + ["model"])
    counts = present.groupby(fkey)["model"].nunique().reset_index(name="n_models")
    rows = []
    for season, g in counts.groupby("season"):
        total = len(g)
        complete = int((g["n_models"] == len(models)).sum())
        rows.append({
            "season": season,
            "models_required": len(models),
            "total_forecast_keys": total,
            "common_forecast_keys": complete,
            "dropped_forecast_keys": total - complete,
            "common_fraction": complete / total if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


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
    common_metrics = common_mask_metrics(metrics)
    common_leaderboard = _season_leaderboard(common_metrics)
    common_summary = _multiseason_summary(common_leaderboard, common_metrics)
    common_report = common_mask_report(metrics)
    phase = _strat(metrics, "phase_origin")
    horizon = _strat(metrics, "horizon")

    leaderboard.to_csv(os.path.join(RES, "season_leaderboard.csv"), index=False)
    summary.to_csv(os.path.join(RES, "multiseason_summary.csv"), index=False)
    common_leaderboard.to_csv(os.path.join(RES, "season_leaderboard_common.csv"), index=False)
    common_summary.to_csv(os.path.join(RES, "multiseason_common_summary.csv"), index=False)
    common_report.to_csv(os.path.join(RES, "common_mask_report.csv"), index=False)
    phase.to_csv(os.path.join(RES, "phase_by_season.csv"), index=False)
    horizon.to_csv(os.path.join(RES, "horizon_by_season.csv"), index=False)

    print("=== multiseason summary (season-unweighted WIS; lower=better) ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== COMMON-MASK multiseason summary (paired; lower=better) ===")
    print(common_summary.round(3).to_string(index=False))
    print("\n=== common-mask accounting ===")
    print(common_report.round(3).to_string(index=False))
    return {"leaderboard": leaderboard, "summary": summary,
            "common_leaderboard": common_leaderboard,
            "common_summary": common_summary,
            "common_report": common_report,
            "phase": phase, "horizon": horizon}


if __name__ == "__main__":
    run()

"""Uncertainty & significance for the multi-season leaderboard.

Two complementary tools, both operating on the matched per-forecast WIS of the
assembled frame (base + ensembles + hybrid):

* **Block bootstrap CIs** clustered by ``(forecast_date, location)``. Forecast
  losses are strongly dependent within a forecast week and within a location
  (overlapping horizons, spatial correlation), so an i.i.d. bootstrap would be
  anticonservative. We resample whole ``(forecast_date, location)`` clusters with
  replacement and recompute the mean WIS difference ``focal - other``; the 2.5/97.5
  percentiles give a CI and the sign-change frequency a two-sided bootstrap p.

* **Pairwise Diebold-Mariano with multiple-comparison control.** The HLN-corrected
  DM test (:func:`evaluation.dm_test.dm_test`) is run for every model pair; raw
  p-values are then adjusted across the family by Holm (FWER) and
  Benjamini-Hochberg (FDR) so a leaderboard-wide "significantly better" claim is
  honest about the number of comparisons made.

Writes ``tables/bootstrap_ci.csv`` and ``tables/multiseason_dm_adjusted.csv``.
The ``focal`` model defaults to the season-unweighted WIS leader, so the headline
answers "is the leader *significantly* ahead of the field?" — and if the hybrid is
not the leader, that is what the table shows.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dm_test import dm_test  # noqa: E402
from evaluation.ensembles import LONG_PATH  # noqa: E402
from evaluation.run_multiseason import assemble_all  # noqa: E402
from evaluation.ensembles import per_forecast_metrics  # noqa: E402

HERE = os.path.dirname(__file__)
TAB = os.path.abspath(os.path.join(HERE, "..", "tables"))

_MATCH = ["season", "forecast_date", "location", "horizon", "reference_date"]


def _matched_wis(metrics: pd.DataFrame) -> pd.DataFrame:
    """Wide WIS table: one row per forecast, one column per model."""
    return metrics.pivot_table(index=_MATCH, columns="model", values="wis")


def _leader(metrics: pd.DataFrame) -> str:
    unw = (metrics.groupby(["model", "season"])["wis"].mean()
           .groupby("model").mean())
    return str(unw.idxmin())


def block_bootstrap_ci(metrics: pd.DataFrame, focal: str | None = None,
                       n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Clustered block-bootstrap CI on mean WIS difference ``focal - other``.

    Resamples whole ``(forecast_date, location)`` clusters with replacement. The
    resampled mean of ``focal - m`` equals ``sum(cluster diff-sums) /
    sum(cluster counts)``, so we precompute per-cluster sums/counts once
    (``np.bincount``) and bootstrap on those aggregates — exact, and fast enough
    for the full panel (the naive per-iteration concat does not scale and is what
    crashed the first full run).
    """
    focal = focal or _leader(metrics)
    wide = _matched_wis(metrics).reset_index()
    codes, uniques = pd.factorize(
        pd.Series(list(zip(wide["forecast_date"], wide["location"]))))
    n_c = len(uniques)

    models = [m for m in metrics["model"].unique() if m != focal]
    focal_arr = wide[focal].to_numpy(dtype=float)

    # Per-cluster diff-sum and valid-count for each model (NaN-skipping, matching
    # a plain mean over matched forecasts).
    csum, ccnt, point = {}, {}, {}
    for m in models:
        d = focal_arr - wide[m].to_numpy(dtype=float)
        valid = ~np.isnan(d)
        csum[m] = np.bincount(codes, weights=np.where(valid, d, 0.0), minlength=n_c)
        ccnt[m] = np.bincount(codes, weights=valid.astype(float), minlength=n_c)
        tot = ccnt[m].sum()
        point[m] = float(csum[m].sum() / tot) if tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        pick = rng.integers(0, n_c, size=n_c)
        for m in models:
            denom = ccnt[m][pick].sum()
            boot[m][b] = csum[m][pick].sum() / denom if denom > 0 else np.nan

    rows = []
    for m in models:
        d = boot[m]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        frac_neg = float(np.mean(d < 0))            # two-sided bootstrap p
        p = 2.0 * min(frac_neg, 1.0 - frac_neg)
        rows.append({"focal": focal, "vs": m, "mean_wis_diff": point[m],
                     "ci_low": float(lo), "ci_high": float(hi),
                     "p_boot": min(1.0, p), "focal_better": point[m] < 0})
    return pd.DataFrame(rows).sort_values("mean_wis_diff").reset_index(drop=True)


def pairwise_dm_adjusted(metrics: pd.DataFrame, method: str = "holm") -> pd.DataFrame:
    """All-pairs HLN-DM with Holm + BH multiple-comparison adjustment."""
    # dm_test matches on [location, forecast_date, horizon]; forecast_date encodes season.
    df = metrics.rename(columns={})  # already has the needed columns
    models = sorted(metrics["model"].unique())
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            r = dm_test(df[df["model"] == a], df[df["model"] == b], metric="wis")
            rows.append({"a": a, "b": b, "n": r["n"], "mean_diff_a_minus_b": r["mean_diff"],
                         "stat": r["stat"], "p": r["p"]})
    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    out["p_holm"], out["p_bh"] = np.nan, np.nan
    if valid.any():
        out.loc[valid, "p_holm"] = multipletests(out.loc[valid, "p"], method="holm")[1]
        out.loc[valid, "p_bh"] = multipletests(out.loc[valid, "p"], method="fdr_bh")[1]
    return out.sort_values("p").reset_index(drop=True)


def run(path: str = LONG_PATH, focal: str | None = None,
        n_boot: int = 1000, seed: int = 0) -> dict:
    os.makedirs(TAB, exist_ok=True)
    metrics = per_forecast_metrics(assemble_all(path))
    focal = focal or _leader(metrics)

    ci = block_bootstrap_ci(metrics, focal=focal, n_boot=n_boot, seed=seed)
    dm = pairwise_dm_adjusted(metrics)
    ci.to_csv(os.path.join(TAB, "bootstrap_ci.csv"), index=False)
    dm.to_csv(os.path.join(TAB, "multiseason_dm_adjusted.csv"), index=False)

    print(f"=== block-bootstrap CI: focal = {focal} (mean WIS diff focal - other) ===")
    print(ci.round(3).to_string(index=False))
    print(f"\n=== pairwise DM (top rows; Holm/BH adjusted) ===")
    print(dm.head(8).round(4).to_string(index=False))
    return {"ci": ci, "dm": dm, "focal": focal}


if __name__ == "__main__":
    run()

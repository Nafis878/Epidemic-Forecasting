"""Diebold-Mariano significance tests for WIS, with the Harvey-Leybourne-Newbold
small-sample correction.

The DM test (Diebold & Mariano, 1995) asks whether two models' forecast losses
differ significantly. For multi-step forecasts the loss differentials are
autocorrelated, so we use a Bartlett (Newey-West) HAC variance with a lag equal to
the maximum horizon, and apply the Harvey-Leybourne-Newbold (1997) finite-sample
correction with a Student-t reference distribution.

Matched forecasts are keyed on ``(location, forecast_date, horizon)``; differentials
are pooled across the panel. We also report an effect size (Cohen's d) and a 95%
confidence interval on the mean WIS difference for the headline comparisons.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

_KEYS = ["location", "forecast_date", "horizon"]


def _match_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    keys = _KEYS.copy()
    for optional in ("disease", "season", "reference_date"):
        if optional in df_a.columns and optional in df_b.columns:
            keys.insert(0, optional) if optional in ("disease", "season") else keys.append(optional)
    return keys


def _hac_var(d: np.ndarray, lag: int) -> float:
    """Bartlett-kernel HAC variance of the mean of ``d``."""
    n = len(d)
    dm = d - d.mean()
    gamma0 = float(np.dot(dm, dm) / n)
    var = gamma0
    for k in range(1, lag + 1):
        if k >= n:
            break
        cov = float(np.dot(dm[k:], dm[:-k]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return max(var, 1e-12)


def dm_test(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "wis",
            max_h: int = 4) -> dict:
    """Diebold-Mariano (HLN-corrected) test that model A's loss != model B's loss.

    Returns a dict with the statistic, two-sided p-value, mean difference
    (A - B; negative => A better), Cohen's d, and a 95% CI on the mean difference.
    """
    m = df_a.merge(df_b, on=_match_keys(df_a, df_b), suffixes=("_a", "_b"))
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy(dtype=float)
    n = len(d)
    if n < 8:
        return {"n": n, "stat": np.nan, "p": np.nan, "mean_diff": float(np.mean(d)) if n else np.nan,
                "cohens_d": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean_d = float(d.mean())
    var = _hac_var(d, lag=max_h)
    se = np.sqrt(var / n)
    dm_stat = mean_d / se
    # HLN small-sample correction (using h = max_h overlapping steps).
    h = max_h
    factor = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-6))
    dm_corr = dm_stat * factor
    p = 2.0 * stats.t.sf(abs(dm_corr), df=n - 1)
    cohens_d = mean_d / (d.std(ddof=1) + 1e-12)
    ci = 1.96 * se
    return {"n": n, "stat": float(dm_corr), "p": float(p), "mean_diff": mean_d,
            "cohens_d": float(cohens_d), "ci_low": mean_d - ci, "ci_high": mean_d + ci}


def dm_matrix(results: pd.DataFrame, models: Optional[list] = None,
              metric: str = "wis", phase: Optional[str] = None) -> pd.DataFrame:
    """Matrix of DM p-values for every model pair (optionally within one phase)."""
    df = results if phase is None else results[results["phase"] == phase]
    models = models or sorted(df["model"].unique())
    P = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for a, b in combinations(models, 2):
        r = dm_test(df[df["model"] == a], df[df["model"] == b], metric=metric)
        P.loc[a, b] = r["p"]
        P.loc[b, a] = r["p"]
    return P


def headline_report(results: pd.DataFrame, focal: str = "mist_v2",
                    metric: str = "wis", phase: Optional[str] = "Rising") -> pd.DataFrame:
    """``focal`` vs every other model (within ``phase``): mean diff, d, CI, p."""
    df = results if phase is None else results[results["phase"] == phase]
    rows = []
    fa = df[df["model"] == focal]
    for other in sorted(df["model"].unique()):
        if other == focal:
            continue
        r = dm_test(fa, df[df["model"] == other], metric=metric)
        rows.append({"vs": other, "mean_wis_diff": r["mean_diff"], "cohens_d": r["cohens_d"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"], "n": r["n"]})
    return pd.DataFrame(rows).sort_values("mean_wis_diff").reset_index(drop=True)

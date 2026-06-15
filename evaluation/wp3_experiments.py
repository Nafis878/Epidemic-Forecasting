"""WP3 bounded win attempt: can any honest combiner beat the median-of-7 ensemble?

The full run showed the median over all base models (~110.4 season-unweighted WIS)
is the robust winner, and that Hedge performance-weighting cannot beat equal
weight. This script tests a small set of leakage-safe candidates against that
bar, so the "iterate-to-win" decision is made on evidence:

* ``median7``            — median over the 7 base models (the bar).
* ``trimmed7``           — drop the per-level min and max, mean the rest (robust mean).
* ``median7_cqr``        — median7 + leakage-safe CQR recalibration (strictly-prior).
* ``mean_top3_prior``    — per target season, equal-weight the 3 base models with the
                           best *prior-season* WIS (leakage-safe selection).

Whatever wins (if anything, with a real margin) is then promoted into the
codebase; otherwise we pivot to the benchmark framing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.ensembles import _KEY, _wide, load_long, per_forecast_metrics
from evaluation.metrics import interval_coverage, weighted_interval_score
from models.phase_stack import _apply_offsets, _cqr_offsets

BASE7 = ["mist_v2", "tft", "patchtst", "arima", "persistence", "ml", "seasonal_naive"]
SEASON_ORDER = ["2022-23", "2023-24", "2024-25"]


def score_wide(wide: pd.DataFrame) -> dict:
    idx = wide.index.to_frame(index=False)
    levels = wide.columns.to_numpy(dtype=float)
    y = idx["y_true"].to_numpy(dtype=float)
    M = wide.to_numpy(dtype=float)
    idx["wis"] = np.atleast_1d(weighted_interval_score(y, M, levels))
    idx["cov50"] = np.atleast_1d(interval_coverage(y, M, 0.50, levels)).astype(float)
    idx["cov95"] = np.atleast_1d(interval_coverage(y, M, 0.95, levels)).astype(float)
    unw = idx.groupby("season")["wis"].mean().mean()
    return {"wis_unweighted": float(unw), "cov50": float(idx["cov50"].mean()),
            "cov95": float(idx["cov95"].mean())}


def _base_wide(long: pd.DataFrame) -> pd.DataFrame:
    return _wide(long[long["model"].isin(BASE7)])


def median7(bw: pd.DataFrame) -> pd.DataFrame:
    return bw.groupby(level=_KEY).median()


def trimmed7(bw: pd.DataFrame) -> pd.DataFrame:
    def _trim(g: np.ndarray) -> np.ndarray:
        if g.shape[0] <= 2:
            return g.mean(axis=0)
        s = np.sort(g, axis=0)
        return s[1:-1].mean(axis=0)
    return bw.groupby(level=_KEY).apply(lambda d: pd.Series(_trim(d.to_numpy()), index=d.columns))


def median7_cqr(med: pd.DataFrame) -> pd.DataFrame:
    levels = med.columns.to_numpy(dtype=float)
    offs = _cqr_offsets(med, levels)
    idx = med.index.to_frame(index=False)
    M = med.to_numpy(dtype=float)
    for i in range(len(M)):
        kb = (int(idx["horizon"].iloc[i]), pd.Timestamp(idx["forecast_date"].iloc[i]))
        M[i] = _apply_offsets(M[i], levels, offs.get(kb, {}))
    return pd.DataFrame(M, index=med.index, columns=levels)


def mean_top3_prior(long: pd.DataFrame, bw: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Per target season, equal-weight the k base models with best PRIOR-season WIS."""
    pm = per_forecast_metrics(long[long["model"].isin(BASE7)])
    season_wis = pm.groupby(["model", "season"])["wis"].mean().unstack("season")
    parts = []
    for si, season in enumerate(SEASON_ORDER):
        if season not in bw.index.get_level_values("season"):
            continue
        prior = SEASON_ORDER[:si]
        if prior:
            prior_mean = season_wis[prior].mean(axis=1)
            chosen = list(prior_mean.nsmallest(k).index)
        else:
            chosen = BASE7                          # no prior -> use all (median fallback)
        sub = bw[bw.index.get_level_values("season") == season]
        sub = sub[sub.index.get_level_values("model").isin(chosen)]
        parts.append(sub.groupby(level=_KEY).mean())
    return pd.concat(parts)


def run() -> pd.DataFrame:
    long = load_long()
    bw = _base_wide(long)
    med = median7(bw)
    candidates = {
        "median7": med,
        "median7_cqr": median7_cqr(med),
        "mean_top3_prior": mean_top3_prior(long, bw),
        "trimmed7": trimmed7(bw),
    }
    rows = []
    for name, wide in candidates.items():
        s = score_wide(wide)
        rows.append({"combiner": name, **s})
    out = pd.DataFrame(rows).sort_values("wis_unweighted").reset_index(drop=True)
    print("=== WP3 bounded win attempt (bar = median7) ===")
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    run()

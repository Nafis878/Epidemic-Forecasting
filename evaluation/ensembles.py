"""Ensemble baselines built from the persisted base-model quantile dump.

Reads ``results/quantiles_long.parquet`` (see :mod:`evaluation.quantile_dump`)
and produces the three standard epidemic-forecasting ensemble references over a
fixed component set:

* ``ens_mean``   — equal-weight average of component quantiles.
* ``ens_median`` — elementwise median of component quantiles (the FluSight Hub's
  long-standing default ensemble).
* ``ens_perf``   — recent-performance-weighted average, **leakage-safe**: at each
  origin ``t`` the Hedge weights use only each component's mean WIS over
  forecasts whose target week was realised *strictly before* ``t`` (prior seasons
  and earlier origins of the current season); equal weighting until any history
  exists.

The ensemble component set defaults to the five models named in the hybrid design
(:data:`COMPONENTS`) so the hybrid and these ensembles are compared over an
identical base — the "phase-aware" claim only holds if the hybrid beats
``ens_perf`` built from the same components.

Output rows share the schema of ``quantiles_long`` (``model`` in
``{ens_mean, ens_median, ens_perf}``) so they concatenate straight onto the base
frame for unified leaderboards.

Note on the perf weights: realised loss uses the *final* truth at the (already
past) target week, a minor revision-related approximation; the leakage boundary
that matters — never using a forecast whose target is at or after ``t`` — is
enforced by the strict ``reference_date < t`` filter.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluation.metrics import interval_coverage, mae, weighted_interval_score
from models.ensemble import (
    equal_weight, hedge_weights, median_combine, trimmed_mean, weighted_combine,
)

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
LONG_PATH = os.path.join(RES, "quantiles_long.parquet")

# The hybrid stacks the five "real" forecasters (its design set).
COMPONENTS = ["mist_v2", "tft", "patchtst", "arima", "persistence"]
# The ensemble baselines follow hub practice and aggregate ALL base forecasters;
# robust combiners (median, trimmed) tolerate the weak members (ml, seasonal_naive)
# that drag the plain mean down.
ENSEMBLE_COMPONENTS = ["mist_v2", "tft", "patchtst", "arima", "persistence",
                       "ml", "seasonal_naive"]
ETA = 0.05  # Hedge temperature on the WIS scale
ENSEMBLE_NAMES = ("ens_mean", "ens_median", "ens_trimmed", "ens_perf")

_KEY = ["season", "forecast_date", "location", "horizon", "reference_date",
        "y_true", "phase_origin"]


def key_cols(df: pd.DataFrame) -> list[str]:
    """Forecast identity columns, including ``disease`` for multi-disease dumps."""
    return (["disease"] if "disease" in df.columns else []) + _KEY


def load_long(path: str = LONG_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def _wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (forecast x model) with quantile levels as columns."""
    idx = ["model"] + key_cols(long)
    w = long.pivot_table(index=idx, columns="quantile", values="value")
    return w.sort_index(axis=1)


def per_forecast_wis(long: pd.DataFrame) -> pd.DataFrame:
    """Per-forecast WIS for every (model, forecast) — vectorised over the panel."""
    w = _wide(long)
    levels = w.columns.to_numpy(dtype=float)
    y = w.index.get_level_values("y_true").to_numpy(dtype=float)
    wis = weighted_interval_score(y, w.to_numpy(dtype=float), levels)
    out = w.index.to_frame(index=False)
    out["wis"] = np.atleast_1d(wis)
    return out


def per_forecast_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per-forecast WIS / MAE / 50% & 95% coverage for every (model, forecast)."""
    w = _wide(long)
    levels = w.columns.to_numpy(dtype=float)
    M = w.to_numpy(dtype=float)
    y = w.index.get_level_values("y_true").to_numpy(dtype=float)
    out = w.index.to_frame(index=False)
    out["wis"] = np.atleast_1d(weighted_interval_score(y, M, levels))
    out["mae"] = np.atleast_1d(mae(y, M, levels))
    out["cov_50"] = np.atleast_1d(interval_coverage(y, M, 0.50, levels)).astype(float)
    out["cov_95"] = np.atleast_1d(interval_coverage(y, M, 0.95, levels)).astype(float)
    return out


def _perf_weights_by_origin(base_wis: pd.DataFrame, components, eta: float):
    """Map each origin (and disease, if present) -> Hedge weights from prior loss."""
    by_origin: dict[pd.Timestamp, np.ndarray] = {}
    group_cols = ["disease"] if "disease" in base_wis.columns else []
    grouped = base_wis.groupby(group_cols) if group_cols else [((), base_wis)]
    for gkey, sub in grouped:
        origins = sorted(pd.to_datetime(sub["forecast_date"].unique()))
        ref = pd.to_datetime(sub["reference_date"])
        prefix = (gkey,) if group_cols and not isinstance(gkey, tuple) else tuple(gkey)
        for t in origins:
            prior = sub[ref < t]
            key = (*prefix, pd.Timestamp(t)) if group_cols else pd.Timestamp(t)
            if prior.empty:
                by_origin[key] = np.full(len(components), 1.0 / len(components))
                continue
            means = prior.groupby("model")["wis"].mean()
            losses = np.array([means.get(m, np.nan) for m in components])
            by_origin[key] = (np.full(len(components), 1.0 / len(components))
                              if np.isnan(losses).any() else hedge_weights(losses, eta))
    return by_origin


def build_ensembles(long: pd.DataFrame, components=ENSEMBLE_COMPONENTS,
                    eta: float = ETA) -> pd.DataFrame:
    """Return long-format ensemble forecasts (mean / median / trimmed / perf)."""
    base = long[long["model"].isin(components)].copy()
    if base.empty:
        return pd.DataFrame(columns=long.columns)

    base_wis = per_forecast_wis(base)
    weights_by_origin = _perf_weights_by_origin(base_wis, components, eta)

    w = _wide(base)
    levels = w.columns.to_numpy(dtype=float)
    keys = key_cols(base)
    rows = []
    for key, sub in w.groupby(level=keys):
        mat = sub.droplevel(keys)                          # index = model
        present = [m for m in components if m in mat.index]
        if not present:
            continue
        B = mat.loc[present].to_numpy(dtype=float)
        keyd = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        origin_key = ((keyd["disease"], pd.Timestamp(keyd["forecast_date"]))
                      if "disease" in keyd else pd.Timestamp(keyd["forecast_date"]))
        wts = weights_by_origin.get(origin_key)
        if wts is not None and len(present) != len(components):
            wts = np.array([wts[components.index(m)] for m in present])  # realign to present
        combos = {
            "ens_mean": equal_weight(B),
            "ens_median": median_combine(B),
            "ens_trimmed": trimmed_mean(B),
            "ens_perf": weighted_combine(B, wts) if wts is not None else equal_weight(B),
        }
        for name, vec in combos.items():
            for q, val in zip(levels, vec):
                rows.append({**keyd, "model": name, "quantile": float(q), "value": float(val)})
    out = pd.DataFrame(rows)
    return out[long.columns] if not out.empty else out


def main(path: str = LONG_PATH) -> pd.DataFrame:
    long = load_long(path)
    ens = build_ensembles(long)
    print(f"built {ens['model'].nunique()} ensembles over {COMPONENTS} "
          f"({len(ens)} rows) from {len(long)} base rows")
    return ens


if __name__ == "__main__":
    main()

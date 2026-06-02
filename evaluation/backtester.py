"""Rolling-origin (time-series) back-testing for MIST-Transformer.

The back-tester walks a list of ``forecast_dates``. At each origin it asks the
versioned store for the data that was knowable *as of that date* and hands only
that slice to the model. This is the single chokepoint that enforces the
no-leakage rule:

    history = store.get_vintage(signal, location, forecast_date)   # <- only path

The model never touches the store, so it is structurally impossible for it to
read future-issued data. Forecasts are then scored against the *final* observed
values using :mod:`evaluation.metrics`.

A "model" is any object exposing::

    predict(history, forecast_date, horizons, quantiles) -> DataFrame
        # columns: [horizon, reference_date, quantile, value]
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import (
    DEFAULT_QUANTILES,
    interval_coverage,
    mae,
    weighted_interval_score,
)
from features.versioned_store import VersionedStore

# A date far enough in the future that get_vintage returns the final revision.
_FINAL_AS_OF = "2100-01-01"


def _final_truth_map(
    store: VersionedStore, signal: str, location: str, source: Optional[str]
) -> dict:
    """Map ``reference_date -> final observed value`` for scoring targets."""
    truth = store.get_vintage(signal, location, _FINAL_AS_OF, source=source)
    return {
        pd.Timestamp(d): float(v)
        for d, v in zip(truth["reference_date"], truth["value"])
    }


def rolling_origin_backtest(
    store: VersionedStore,
    model,
    forecast_dates: Sequence,
    *,
    signal: str,
    location: str,
    source: Optional[str] = None,
    horizons: Sequence[int] = (1, 2, 3, 4),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    truth_source: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate ``model`` over ``forecast_dates`` with strict vintage inputs.

    Parameters
    ----------
    store:
        The versioned data store.
    model:
        Object with a ``predict(history, forecast_date, horizons, quantiles)``
        method returning long-format quantile forecasts.
    forecast_dates:
        Rolling forecast origins.
    signal, location, source:
        Series selectors passed to ``get_vintage`` for the model's input.
    horizons:
        Steps ahead requested from the model.
    quantiles:
        Quantile levels requested from the model.
    truth_source:
        Source used to look up the *final* observed values for scoring. Defaults
        to ``source`` (score a source against its own final revisions).

    Returns
    -------
    pandas.DataFrame
        One row per (forecast_date, horizon) with the target ``reference_date``,
        the observed ``y_true``, the predicted ``median``, and per-forecast
        ``wis`` / ``mae`` / ``cov_50`` / ``cov_95``. Rows whose target has no
        observed value (not yet realised) are skipped.
    """
    quantiles = list(quantiles)
    truth_map = _final_truth_map(store, signal, location, truth_source or source)

    records = []
    for fdate in forecast_dates:
        fdate = pd.Timestamp(fdate)
        # === the only data access: strictly vintage, no leakage ===
        history = store.get_vintage(signal, location, fdate, source=source)
        if history.empty:
            continue

        preds = model.predict(
            history=history, forecast_date=fdate,
            horizons=horizons, quantiles=quantiles,
        )

        for horizon, grp in preds.groupby("horizon"):
            grp = grp.sort_values("quantile")
            q_levels = grp["quantile"].to_numpy(dtype=float)
            q_vals = grp["value"].to_numpy(dtype=float)
            target_date = pd.Timestamp(grp["reference_date"].iloc[0])

            y = truth_map.get(target_date)
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue  # target not yet observed -> cannot score

            median = float(np.interp(0.5, q_levels, q_vals))
            records.append(
                {
                    "forecast_date": fdate,
                    "horizon": int(horizon),
                    "reference_date": target_date,
                    "location": location,
                    "source": source,
                    "y_true": y,
                    "median": median,
                    "wis": float(weighted_interval_score(y, q_vals, q_levels)),
                    "mae": float(mae(y, q_vals, q_levels)),
                    "cov_50": bool(interval_coverage(y, q_vals, 0.50, q_levels)),
                    "cov_95": bool(interval_coverage(y, q_vals, 0.95, q_levels)),
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "forecast_date", "horizon", "reference_date", "location", "source",
            "y_true", "median", "wis", "mae", "cov_50", "cov_95",
        ],
    )


def summarize(results: pd.DataFrame, by: Sequence[str] = ("horizon",)) -> pd.DataFrame:
    """Aggregate back-test results (mean WIS/MAE, empirical coverage)."""
    if results.empty:
        return results
    agg = (
        results.groupby(list(by))
        .agg(
            n=("wis", "size"),
            wis=("wis", "mean"),
            mae=("mae", "mean"),
            coverage_50=("cov_50", "mean"),
            coverage_95=("cov_95", "mean"),
        )
        .reset_index()
    )
    return agg

"""Simple baseline forecaster used to exercise the evaluation pipeline.

``PersistenceQuantileModel`` is a classic epidemic-forecasting baseline: the
median forecast is the most recent observed value (persistence), and forecast
uncertainty grows with the horizon, calibrated from the recent week-over-week
change in the series. Quantiles are produced from a Normal distribution using
the standard library (no SciPy dependency) and clipped at zero (counts are
non-negative).

It deliberately depends only on the ``history`` handed to it by the backtester,
which is a strict vintage view from :meth:`VersionedStore.get_vintage` — the
model has no access to the store and therefore cannot leak future data.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import DEFAULT_QUANTILES


class PersistenceQuantileModel:
    """Persistence median with horizon-scaled Normal quantiles.

    Parameters
    ----------
    cadence_days:
        Spacing between consecutive observations (7 for weekly data). Target
        dates are ``last_observed_date + horizon * cadence_days``.
    min_sigma:
        Floor on the 1-step standard deviation, so degenerate flat history still
        yields a non-trivial predictive interval.
    """

    def __init__(self, cadence_days: int = 7, min_sigma: float = 1.0) -> None:
        self.cadence_days = cadence_days
        self.min_sigma = min_sigma

    def predict(
        self,
        history: pd.DataFrame,
        forecast_date,
        horizons: Sequence[int] = (1, 2, 3, 4),
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Return long-format quantile forecasts.

        Parameters
        ----------
        history:
            Vintage truth as of ``forecast_date`` with columns
            ``reference_date`` and ``value`` (as returned by ``get_vintage``).
        horizons:
            Steps ahead to forecast.
        quantiles:
            Quantile levels to emit.

        Returns
        -------
        pandas.DataFrame
            Columns ``[horizon, reference_date, quantile, value]``.
        """
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        last_value = float(hist["value"].iloc[-1])

        # 1-step sigma from recent week-over-week changes (last ~8 diffs).
        diffs = hist["value"].astype(float).diff().dropna().to_numpy()
        recent = diffs[-8:] if len(diffs) >= 1 else np.array([0.0])
        sigma1 = max(float(np.std(recent)) if recent.size else 0.0, self.min_sigma)

        rows = []
        for h in horizons:
            target_date = last_date + pd.Timedelta(days=self.cadence_days * h)
            # Random-walk variance grows linearly with horizon -> sd ~ sqrt(h).
            sigma_h = sigma1 * np.sqrt(h)
            nd = NormalDist(mu=last_value, sigma=sigma_h)
            for q in quantiles:
                val = max(0.0, nd.inv_cdf(q))
                rows.append(
                    {"horizon": h, "reference_date": target_date,
                     "quantile": float(q), "value": val}
                )
        return pd.DataFrame(rows)

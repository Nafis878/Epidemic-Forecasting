"""Simple baseline forecasters used to exercise the evaluation pipeline.

``PersistenceQuantileModel`` is a classic epidemic-forecasting baseline: the
median forecast is the most recent observed value (persistence), and forecast
uncertainty grows with the horizon, calibrated from the recent week-over-week
change in the series. ``SeasonalNaiveQuantileModel`` is the other canonical
naive reference: the median forecast for a target week is the value observed for
the same calendar week one year earlier (52-week lag), falling back to
persistence when no seasonal lag is available in the vintage history. Quantiles
are produced from a Normal distribution using the standard library (no SciPy
dependency) and clipped at zero (counts are non-negative).

Both deliberately depend only on the ``history`` handed to them by the
backtester, which is a strict vintage view from
:meth:`VersionedStore.get_vintage` — the model has no access to the store and
therefore cannot leak future data.
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


class SeasonalNaiveQuantileModel:
    """Seasonal-naive median (value 52 weeks earlier) with Normal quantiles.

    The standard FluSight "seasonal naive" reference: the point forecast for a
    target week is the value observed for the same week of the previous season.
    When the vintage history does not yet reach back a full season (early data),
    it falls back to persistence (last observed value). Predictive spread grows
    with the horizon, scaled from recent week-over-week variability — identical
    machinery to :class:`PersistenceQuantileModel` so the two naive baselines are
    comparable.

    Parameters
    ----------
    season_lag_weeks:
        Number of weeks defining one season (52 for weekly influenza data).
    cadence_days, min_sigma:
        As in :class:`PersistenceQuantileModel`.
    lag_tolerance_days:
        Half-width of the date window used to match the seasonal-lag observation
        (the prior season's weeks rarely align to the exact calendar day).
    """

    def __init__(self, season_lag_weeks: int = 52, cadence_days: int = 7,
                 min_sigma: float = 1.0, lag_tolerance_days: int = 10) -> None:
        self.season_lag_weeks = season_lag_weeks
        self.cadence_days = cadence_days
        self.min_sigma = min_sigma
        self.lag_tolerance_days = lag_tolerance_days

    def predict(
        self,
        history: pd.DataFrame,
        forecast_date,
        horizons: Sequence[int] = (1, 2, 3, 4),
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        last_value = float(hist["value"].iloc[-1])

        dates = pd.to_datetime(hist["reference_date"]).to_numpy()
        values = hist["value"].astype(float).to_numpy()

        diffs = hist["value"].astype(float).diff().dropna().to_numpy()
        recent = diffs[-8:] if len(diffs) >= 1 else np.array([0.0])
        sigma1 = max(float(np.std(recent)) if recent.size else 0.0, self.min_sigma)

        lag = pd.Timedelta(days=self.season_lag_weeks * 7)
        tol = pd.Timedelta(days=self.lag_tolerance_days)

        rows = []
        for h in horizons:
            target_date = last_date + pd.Timedelta(days=self.cadence_days * h)
            # Median: the prior-season value nearest to (target_date - 52w).
            anchor = target_date - lag
            within = np.abs(dates - np.datetime64(anchor)) <= np.timedelta64(tol)
            center = float(values[within][np.argmin(np.abs(dates[within] - np.datetime64(anchor)))]) \
                if within.any() else last_value
            sigma_h = sigma1 * np.sqrt(h)
            nd = NormalDist(mu=center, sigma=sigma_h)
            for q in quantiles:
                rows.append(
                    {"horizon": h, "reference_date": target_date,
                     "quantile": float(q), "value": max(0.0, nd.inv_cdf(q))}
                )
        return pd.DataFrame(rows)

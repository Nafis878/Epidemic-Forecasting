"""Statistical baseline: an ARIMA-based quantile forecaster.

We use ``statsmodels`` ARIMA (Prophet is not installed and is awkward to build on
Windows; ARIMA is the standard statistical baseline for this kind of weekly
count series). The model is fit to the vintage history handed in by the
back-tester, point forecasts and forecast standard errors are produced per
horizon, and 23 quantiles are generated from a Normal predictive distribution
``N(mean_h, se_h)`` (clipped at zero — admissions are non-negative).

If ARIMA fails to fit (too little history, non-convergence) the model falls back
to a random-walk Normal whose spread grows with ``sqrt(h)``, so ``predict`` never
raises inside a back-test.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.metrics import DEFAULT_QUANTILES

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.arima.model import ARIMA


class ARIMAQuantileModel:
    """ARIMA point forecast wrapped in a Normal predictive distribution.

    Parameters
    ----------
    order:
        ARIMA ``(p, d, q)`` order. Default ``(2, 1, 1)`` — a differenced model
        suited to trending weekly counts.
    cadence_days:
        Observation spacing (7 for weekly). Targets are
        ``last_observed_date + horizon * cadence_days``.
    min_train:
        Minimum observations before attempting an ARIMA fit; below this the
        random-walk fallback is used.
    min_sigma:
        Floor on the predictive standard deviation.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 1),
        cadence_days: int = 7,
        min_train: int = 12,
        min_sigma: float = 1.0,
    ) -> None:
        self.order = order
        self.cadence_days = cadence_days
        self.min_train = min_train
        self.min_sigma = min_sigma

    def _forecast_mean_se(self, y: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-step (mean, se). Falls back to a random walk on failure."""
        if len(y) >= self.min_train:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(
                        y, order=self.order,
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit()
                    fc = fit.get_forecast(steps=steps)
                    mean = np.asarray(fc.predicted_mean, dtype=float)
                    se = np.asarray(fc.se_mean, dtype=float)
                if np.all(np.isfinite(mean)) and np.all(np.isfinite(se)):
                    return mean, np.maximum(se, self.min_sigma)
            except Exception:
                pass
        # Random-walk fallback: flat mean, sigma ~ recent diff std * sqrt(h).
        last = float(y[-1])
        diffs = np.diff(y)
        sigma1 = max(float(np.std(diffs[-8:])) if diffs.size else 0.0, self.min_sigma)
        mean = np.full(steps, last)
        se = sigma1 * np.sqrt(np.arange(1, steps + 1))
        return mean, np.maximum(se, self.min_sigma)

    def predict(
        self,
        history: pd.DataFrame,
        forecast_date,
        horizons: Sequence[int] = (1, 2, 3, 4),
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        y = hist["value"].astype(float).to_numpy()

        steps = max(horizons)
        mean, se = self._forecast_mean_se(y, steps)

        rows = []
        for h in horizons:
            mu, s = float(mean[h - 1]), float(se[h - 1])
            nd = NormalDist(mu=mu, sigma=max(s, self.min_sigma))
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            for q in quantiles:
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": max(0.0, nd.inv_cdf(q))})
        return pd.DataFrame(rows)

"""Mechanistic baseline: a standard SEIR model calibrated on vintage history.

A classic four-compartment **SEIR** model (Susceptible -> Exposed -> Infectious
-> Recovered) is fit to the recent vintage history handed in by the back-tester,
then integrated forward to produce horizon forecasts. Weekly *incident
hospital admissions* are modelled as a fixed fraction ``rho`` of new infections
(the S->E flow) accumulated over each week.

Dynamics (daily rates, population ``N``)::

    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E - gamma * I
    dR/dt =  gamma * I
    dC/dt =  beta * S * I / N            # cumulative new infections

with ``beta = R0 * gamma``. The incubation/recovery rates are fixed to
flu-like values (``sigma = 1/3``/day, ``gamma = 1/5``/day); the calibrated
parameters are ``theta = [R0, E0, I0, rho]``, fit by non-linear least squares on
the most recent ``fit_window`` weeks. Forecast uncertainty comes from the
in-sample residual spread, widened with ``sqrt(horizon)``.

Calibration uses *only* the vintage history passed in, so the back-tester's
no-leakage guarantee is preserved. A failed fit falls back to a random-walk
Normal, so ``predict`` never raises.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from evaluation.metrics import DEFAULT_QUANTILES

# Approximate populations for the locations we back-test (FIPS + abbrev keys).
POPULATION = {
    "US": 3.30e8, "USA": 3.30e8, "nat": 3.30e8,
    "06": 3.90e7, "CA": 3.90e7,   # California
    "36": 1.95e7, "NY": 1.95e7,   # New York
}
DEFAULT_POP = 1.0e7


def _seir_weekly_incidence(theta, N, n_weeks, sigma, gamma):
    """Integrate the SEIR ODE and return modelled weekly hospital admissions.

    ``theta = [R0, E0, I0, rho]``. Returns an array of length ``n_weeks``.
    """
    R0, E0, I0, rho = theta
    beta = R0 * gamma
    S0 = max(N - E0 - I0, 1.0)
    y0 = [S0, E0, I0, 0.0, 0.0]  # S, E, I, R, C(cumulative infections)

    def rhs(t, y):
        S, E, I, R, C = y
        new_inf = beta * S * I / N
        return [-new_inf, new_inf - sigma * E, sigma * E - gamma * I, gamma * I, new_inf]

    n_days = n_weeks * 7
    t_eval = np.arange(0, n_days + 1, 7)  # week boundaries (n_weeks + 1 points)
    sol = solve_ivp(rhs, (0, n_days), y0, t_eval=t_eval, method="RK45",
                    rtol=1e-5, atol=1e-2)
    if not sol.success or sol.y.shape[1] < n_weeks + 1:
        return np.full(n_weeks, np.nan)
    cumulative = sol.y[4]
    weekly_new_infections = np.diff(cumulative)        # length n_weeks
    return rho * weekly_new_infections


class SEIRModel:
    """SEIR forecaster calibrated per-origin on recent vintage history."""

    def __init__(
        self,
        cadence_days: int = 7,
        sigma: float = 1.0 / 3.0,
        gamma: float = 1.0 / 5.0,
        fit_window: int = 20,
        min_train: int = 8,
        max_nfev: int = 60,
        min_sigma: float = 1.0,
    ) -> None:
        self.cadence_days = cadence_days
        self.sigma = sigma
        self.gamma = gamma
        self.fit_window = fit_window
        self.min_train = min_train
        self.max_nfev = max_nfev
        self.min_sigma = min_sigma

    # ---------------------------------------------------------------- internals
    def _calibrate(self, obs: np.ndarray, N: float):
        """Fit theta to ``obs`` (recent weekly admissions). Returns (theta, resid_std)."""
        L = len(obs)
        scale = max(float(np.mean(obs)), 1.0)
        rho0 = 0.005
        # Rough initial infectious pool consistent with observed admissions.
        I0 = max(scale / (rho0 * self.gamma * 7.0), 1.0)
        x0 = np.array([1.3, I0, I0, rho0])
        lower = np.array([0.6, 0.0, 0.0, 1e-5])
        upper = np.array([4.0, 0.05 * N, 0.05 * N, 0.5])

        def resid(theta):
            modelled = _seir_weekly_incidence(theta, N, L, self.sigma, self.gamma)
            if not np.all(np.isfinite(modelled)):
                return np.full(L, 1e6)
            return (modelled - obs) / scale  # scaled residuals

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = least_squares(resid, x0, bounds=(lower, upper),
                                max_nfev=self.max_nfev, method="trf")
        theta = res.x
        fitted = _seir_weekly_incidence(theta, N, L, self.sigma, self.gamma)
        resid_std = float(np.std(obs - fitted)) if np.all(np.isfinite(fitted)) else np.nan
        return theta, resid_std

    def _fallback(self, y, last_date, horizons, quantiles):
        last = float(y[-1])
        diffs = np.diff(y)
        sigma1 = max(float(np.std(diffs[-8:])) if diffs.size else 0.0, self.min_sigma)
        rows = []
        for h in horizons:
            nd = NormalDist(mu=last, sigma=sigma1 * np.sqrt(h))
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            for q in quantiles:
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": max(0.0, nd.inv_cdf(q))})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------- predict
    def predict(
        self,
        history: pd.DataFrame,
        forecast_date,
        horizons: Sequence[int] = (1, 2, 3, 4),
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        location: str = None,
    ) -> pd.DataFrame:
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        y = hist["value"].astype(float).to_numpy()

        loc = location if location is not None else (
            hist["location"].iloc[-1] if "location" in hist.columns else None
        )
        N = POPULATION.get(str(loc), DEFAULT_POP)

        if len(y) < self.min_train:
            return self._fallback(y, last_date, horizons, quantiles)

        obs = y[-self.fit_window:]
        L = len(obs)
        try:
            theta, resid_std = self._calibrate(obs, N)
            total_weeks = L + max(horizons)
            modelled = _seir_weekly_incidence(theta, N, total_weeks, self.sigma, self.gamma)
            if not np.all(np.isfinite(modelled)) or not np.isfinite(resid_std):
                return self._fallback(y, last_date, horizons, quantiles)
        except Exception:
            return self._fallback(y, last_date, horizons, quantiles)

        sigma_base = max(resid_std, self.min_sigma)
        rows = []
        for h in horizons:
            mu = float(modelled[L - 1 + h])           # week (L-1) is last_obs
            s = sigma_base * np.sqrt(h)
            nd = NormalDist(mu=max(mu, 0.0), sigma=s)
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            for q in quantiles:
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": max(0.0, nd.inv_cdf(q))})
        return pd.DataFrame(rows)

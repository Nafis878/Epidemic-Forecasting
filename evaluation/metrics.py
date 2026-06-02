"""Probabilistic forecast scoring for MIST-Transformer.

Implements the metrics used by the CDC FluSight / COVID-19 Forecast Hub:

* **Weighted Interval Score (WIS)** — the primary metric. A quantile-based
  approximation to the CRPS (Bracher et al., 2021). For the standard 23-quantile
  set it is built from the median plus 11 symmetric central prediction intervals.
* **MAE** — absolute error of the median (0.5) forecast.
* **Interval coverage** — empirical coverage of the nominal 50% ([0.25, 0.75])
  and 95% ([0.025, 0.975]) central prediction intervals.

All functions accept a single forecast (1-D quantile vector) or a batch
(2-D array, one row per forecast). Quantile values are linearly interpolated
across the supplied levels, so a non-standard quantile set still works.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# The 23-quantile forecast set mandated by the FluSight / COVID-19 Forecast Hub.
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.95, 0.975, 0.99,
)

ArrayLike = Union[Sequence[float], np.ndarray]


def _alphas_from_levels(levels: np.ndarray) -> np.ndarray:
    """Central-interval alphas implied by the symmetric quantile levels (<0.5)."""
    lowers = levels[levels < 0.5 - 1e-9]
    # alpha for a (alpha/2, 1-alpha/2) interval is 2 * lower-quantile-level.
    return np.sort(2.0 * lowers)


def _as_2d(quantile_preds: ArrayLike) -> tuple[np.ndarray, bool]:
    """Return (2-D preds, was_1d)."""
    arr = np.asarray(quantile_preds, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _q(preds2d: np.ndarray, levels: np.ndarray, tau: float) -> np.ndarray:
    """Value at quantile level ``tau`` for each row, via linear interpolation."""
    return np.array([np.interp(tau, levels, row) for row in preds2d])


def _interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    alpha: float) -> np.ndarray:
    """Interval score for the central (1-alpha) prediction interval."""
    below = np.where(y < lower, lower - y, 0.0)
    above = np.where(y > upper, y - upper, 0.0)
    return (upper - lower) + (2.0 / alpha) * (below + above)


def weighted_interval_score(
    y_true: Union[float, ArrayLike],
    quantile_preds: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
) -> Union[float, np.ndarray]:
    """Weighted Interval Score (lower is better).

    ``WIS = 1/(K + 1/2) * [ 1/2 * |y - m| + sum_k (alpha_k/2) * IS_{alpha_k} ]``

    Parameters
    ----------
    y_true:
        Observed value(s). Scalar for one forecast, length-n for a batch.
    quantile_preds:
        Predicted quantile values. 1-D (one forecast) or 2-D ``(n, L)``.
    quantile_levels:
        The quantile levels the predictions correspond to (sorted not required).

    Returns
    -------
    float or numpy.ndarray
        Scalar if a single forecast was supplied, else an array of length n.
    """
    levels = np.asarray(quantile_levels, dtype=float)
    order = np.argsort(levels)
    levels = levels[order]

    preds, was_1d = _as_2d(quantile_preds)
    preds = preds[:, order]
    y = np.atleast_1d(np.asarray(y_true, dtype=float))

    alphas = _alphas_from_levels(levels)
    K = len(alphas)

    median = _q(preds, levels, 0.5)
    total = 0.5 * np.abs(y - median)
    for alpha in alphas:
        lower = _q(preds, levels, alpha / 2.0)
        upper = _q(preds, levels, 1.0 - alpha / 2.0)
        total += (alpha / 2.0) * _interval_score(y, lower, upper, alpha)
    wis = total / (K + 0.5)

    return float(wis[0]) if was_1d else wis


def mae(
    y_true: Union[float, ArrayLike],
    quantile_preds: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
) -> Union[float, np.ndarray]:
    """Absolute error of the median (0.5 quantile) forecast."""
    levels = np.asarray(quantile_levels, dtype=float)
    order = np.argsort(levels)
    preds, was_1d = _as_2d(quantile_preds)
    median = _q(preds[:, order], levels[order], 0.5)
    y = np.atleast_1d(np.asarray(y_true, dtype=float))
    err = np.abs(y - median)
    return float(err[0]) if was_1d else err


def interval_coverage(
    y_true: Union[float, ArrayLike],
    quantile_preds: ArrayLike,
    level: float,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
) -> Union[bool, np.ndarray]:
    """Whether each observation falls inside the nominal central interval.

    ``level`` is the nominal coverage (e.g. 0.50 or 0.95); the interval is
    ``[(1-level)/2, 1-(1-level)/2]``. Returns a boolean (single) or array.
    """
    levels = np.asarray(quantile_levels, dtype=float)
    order = np.argsort(levels)
    levels = levels[order]
    preds, was_1d = _as_2d(quantile_preds)
    preds = preds[:, order]
    y = np.atleast_1d(np.asarray(y_true, dtype=float))

    alpha = 1.0 - level
    lower = _q(preds, levels, alpha / 2.0)
    upper = _q(preds, levels, 1.0 - alpha / 2.0)
    inside = (y >= lower) & (y <= upper)
    return bool(inside[0]) if was_1d else inside


def coverage_50(y_true, quantile_preds, quantile_levels=DEFAULT_QUANTILES):
    """Empirical coverage of the nominal 50% interval (mean over observations)."""
    return float(np.mean(interval_coverage(y_true, quantile_preds, 0.50, quantile_levels)))


def coverage_95(y_true, quantile_preds, quantile_levels=DEFAULT_QUANTILES):
    """Empirical coverage of the nominal 95% interval (mean over observations)."""
    return float(np.mean(interval_coverage(y_true, quantile_preds, 0.95, quantile_levels)))


def score_summary(
    y_true: ArrayLike,
    quantile_preds: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
) -> dict:
    """Aggregate all metrics over a batch of forecasts into a single dict."""
    y_true = np.asarray(y_true, dtype=float)
    preds, _ = _as_2d(quantile_preds)
    return {
        "n": int(len(y_true)),
        "wis": float(np.mean(weighted_interval_score(y_true, preds, quantile_levels))),
        "mae": float(np.mean(mae(y_true, preds, quantile_levels))),
        "coverage_50": coverage_50(y_true, preds, quantile_levels),
        "coverage_95": coverage_95(y_true, preds, quantile_levels),
    }

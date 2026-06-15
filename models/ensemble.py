"""Pure quantile-combination primitives for ensembles and the hybrid.

These are deliberately side-effect-free array functions so they can be unit
tested in isolation and reused by both the ensemble baselines
(:mod:`evaluation.ensembles`) and the phase-stacked hybrid
(:mod:`models.phase_stack`).

All combiners take a base matrix ``B`` of shape ``(M, Q)`` — ``M`` component
models, ``Q`` quantile levels, each row a model's quantile vector **already
sorted by level** — and return a length-``Q`` combined vector. Because each input
row is monotone non-decreasing, an elementwise weighted average or median is also
monotone (a convex/order-statistic combination of sorted vectors stays sorted),
so the combined quantiles never cross. A final ``np.sort`` is applied as a cheap
safety net against floating-point noise.
"""

from __future__ import annotations

import numpy as np


def equal_weight(base: np.ndarray) -> np.ndarray:
    """Equal-weight mean of component quantiles at each level."""
    return np.sort(np.asarray(base, dtype=float).mean(axis=0))


def median_combine(base: np.ndarray) -> np.ndarray:
    """Elementwise median of component quantiles at each level."""
    return np.sort(np.median(np.asarray(base, dtype=float), axis=0))


def trimmed_mean(base: np.ndarray) -> np.ndarray:
    """Per-level trimmed mean: drop the min and max component, average the rest.

    A robust aggregator that, unlike the plain mean, tolerates weak/outlying
    component models (it discards the worst at each quantile level) while keeping
    more of the central mass than the median. On the flu benchmark this is the
    single strongest combiner. Falls back to the mean with <=2 components.
    """
    b = np.asarray(base, dtype=float)
    if b.shape[0] <= 2:
        return np.sort(b.mean(axis=0))
    s = np.sort(b, axis=0)                  # sort components within each level
    return np.sort(s[1:-1].mean(axis=0))    # drop min & max, average the middle


def weighted_combine(base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Convex weighted average of component quantiles (weights need not sum to 1)."""
    base = np.asarray(base, dtype=float)
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    w = np.full(len(base), 1.0 / len(base)) if s <= 0 else w / s
    return np.sort((w[:, None] * base).sum(axis=0))


def hedge_weights(mean_losses: np.ndarray, eta: float) -> np.ndarray:
    """Hedge / exponential weights ``w_m ∝ exp(-eta * meanloss_m)`` (normalised).

    Lower-loss components get exponentially more weight; ``eta -> 0`` recovers
    equal weighting. A stable softmax (subtract the max) avoids overflow.
    """
    z = -float(eta) * np.asarray(mean_losses, dtype=float)
    z = z - np.nanmax(z)
    w = np.exp(np.where(np.isfinite(z), z, -np.inf))
    s = w.sum()
    return np.full(len(w), 1.0 / len(w)) if s <= 0 else w / s

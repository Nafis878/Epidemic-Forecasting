"""Tests for evaluation.metrics (WIS / MAE / coverage)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import (  # noqa: E402
    DEFAULT_QUANTILES,
    coverage_50,
    coverage_95,
    mae,
    weighted_interval_score,
)


def _pinball(y, q_val, tau):
    """Standard pinball / quantile loss."""
    return (tau * (y - q_val)) if y >= q_val else ((1 - tau) * (q_val - y))


def test_there_are_23_quantiles():
    assert len(DEFAULT_QUANTILES) == 23
    assert 0.5 in DEFAULT_QUANTILES


def test_perfect_forecast_scores_zero():
    """All quantiles equal to y => WIS = 0 and MAE = 0."""
    y = 42.0
    preds = np.full(len(DEFAULT_QUANTILES), y)
    assert weighted_interval_score(y, preds) == pytest.approx(0.0)
    assert mae(y, preds) == pytest.approx(0.0)


def test_degenerate_forecast_wis_equals_mae():
    """A point mass (all quantiles = m) makes WIS reduce to |y - m|."""
    m = 10.0
    y = 17.0
    preds = np.full(len(DEFAULT_QUANTILES), m)
    assert weighted_interval_score(y, preds) == pytest.approx(abs(y - m))


def test_wis_matches_pinball_identity():
    """WIS equals (2 / (2K+1)) * sum of pinball losses over all quantile levels."""
    rng = np.random.default_rng(0)
    levels = np.array(DEFAULT_QUANTILES)
    # A valid (monotone) quantile forecast.
    preds = np.sort(rng.normal(100, 15, size=len(levels)))
    y = 108.0

    wis = weighted_interval_score(y, preds, levels)
    pinball_sum = sum(_pinball(y, q, tau) for q, tau in zip(preds, levels))
    expected = (2.0 / len(levels)) * pinball_sum  # 2K+1 == number of levels
    assert wis == pytest.approx(expected, rel=1e-9)


def test_batch_shapes_and_coverage():
    """Batch input returns per-row WIS; coverage is in [0, 1]."""
    levels = np.array(DEFAULT_QUANTILES)
    rng = np.random.default_rng(1)
    n = 200
    preds = np.sort(rng.normal(50, 10, size=(n, len(levels))), axis=1)
    y = rng.normal(50, 10, size=n)

    wis = weighted_interval_score(y, preds, levels)
    assert wis.shape == (n,)
    assert np.all(wis >= 0)

    c50 = coverage_50(y, preds, levels)
    c95 = coverage_95(y, preds, levels)
    assert 0.0 <= c50 <= 1.0
    assert 0.0 <= c95 <= 1.0
    # 95% interval must cover at least as often as the 50% interval.
    assert c95 >= c50


def test_wider_interval_when_outside_increases_wis():
    """Missing the observation costs more for a tighter (overconfident) interval."""
    levels = np.array(DEFAULT_QUANTILES)
    y = 100.0
    tight = np.linspace(40, 60, len(levels))   # far from y, narrow
    wide = np.linspace(20, 130, len(levels))   # also off-centre but wider/closer
    assert weighted_interval_score(y, tight, levels) > weighted_interval_score(y, wide, levels)

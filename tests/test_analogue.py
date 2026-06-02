"""Phase 3.3 tests: DTW correctness and leakage-safe analogue retrieval."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.analogue import AnaloguePrior, dtw_distance  # noqa: E402


def test_dtw_zero_and_symmetric():
    a = np.array([1.0, 2, 3, 4, 3, 2, 1])
    b = np.array([1.0, 2, 2, 3, 4, 2, 1])
    assert dtw_distance(a, a) == 0.0
    assert abs(dtw_distance(a, b) - dtw_distance(b, a)) < 1e-9


def test_dtw_aligns_time_shift():
    """A time-warped copy is closer than an unrelated sequence."""
    base = np.sin(np.linspace(0, np.pi, 12))
    warped = np.sin(np.linspace(0, np.pi, 12) - 0.2)
    noise = np.linspace(0, 1, 12)
    assert dtw_distance(base, warped) < dtw_distance(base, noise)


def _synthetic_ili(n_seasons=6):
    """Seasonal ILI-like series: a bell wave per season."""
    rows = []
    start = pd.Timestamp("2000-10-07")
    for s in range(n_seasons):
        for w in range(30):
            d = start + pd.Timedelta(weeks=s * 52 + w)
            v = 1.0 + 5.0 * np.exp(-((w - 12) ** 2) / 30.0)
            rows.append({"reference_date": d, "value": v})
    return pd.DataFrame(rows)


def test_analogue_forecast_shape_length():
    ser = _synthetic_ili()
    prior = AnaloguePrior(ser, window=8, horizon=4)
    recent = ser["value"].to_numpy()[40:48]
    shape, dists = prior.forecast_shape(recent, cutoff=ser["reference_date"].iloc[-1], k=3)
    assert shape is not None and len(shape) == 4
    assert dists is not None and len(dists) == 3


def test_analogue_excludes_future_seasons():
    """Windows whose continuation reaches on/after the cutoff are excluded (no leakage)."""
    ser = _synthetic_ili()
    prior = AnaloguePrior(ser, window=8, horizon=4)
    early_cutoff = ser["reference_date"].iloc[60]
    lib = prior._windows_before(early_cutoff)
    assert all(end < early_cutoff for _, _, end in lib)
    # With a very early cutoff there is not enough history -> graceful None.
    shape, _ = prior.forecast_shape(ser["value"].to_numpy()[:8],
                                    cutoff=ser["reference_date"].iloc[10], k=5)
    assert shape is None

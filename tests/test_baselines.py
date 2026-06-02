"""Smoke/regression tests for the statistical and mechanistic baselines.

These run on synthetic in-memory history (no network) and check the forecast
contract the back-tester relies on: 23 monotone, non-negative quantiles per
horizon, with the right target dates.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import DEFAULT_QUANTILES  # noqa: E402
from models.baseline_mechanistic import SEIRModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402


def _history(n=40, loc="US"):
    """Synthetic epidemic-ish weekly curve (rise then fall)."""
    dates = pd.date_range("2023-09-30", periods=n, freq="7D")
    t = np.linspace(0, np.pi, n)
    values = 200 + 1800 * np.sin(t) ** 2  # bell-shaped wave
    return pd.DataFrame({"reference_date": dates, "value": values, "location": loc})


@pytest.mark.parametrize("model", [ARIMAQuantileModel(), SEIRModel(max_nfev=40)])
def test_forecast_contract(model):
    hist = _history()
    last_date = hist["reference_date"].max()
    preds = model.predict(history=hist, forecast_date=last_date + pd.Timedelta(days=4),
                          horizons=(1, 2, 3, 4), quantiles=DEFAULT_QUANTILES)
    assert set(preds["horizon"]) == {1, 2, 3, 4}
    for h, g in preds.groupby("horizon"):
        g = g.sort_values("quantile")
        v = g["value"].to_numpy()
        assert len(v) == 23
        assert np.all(v >= 0)                       # counts are non-negative
        assert np.all(np.diff(v) >= -1e-6)          # monotone non-decreasing
        expected_target = last_date + pd.Timedelta(days=7 * h)
        assert pd.Timestamp(g["reference_date"].iloc[0]) == expected_target


@pytest.mark.parametrize("model", [ARIMAQuantileModel(), SEIRModel()])
def test_short_history_falls_back_without_error(model):
    """Too little history must not raise — the fallback path kicks in."""
    hist = _history(n=5)
    preds = model.predict(history=hist, forecast_date=hist["reference_date"].max(),
                          horizons=(1, 2))
    assert len(preds) == 2 * 23
    assert np.all(preds["value"] >= 0)

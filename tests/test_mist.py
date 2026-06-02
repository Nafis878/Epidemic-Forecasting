"""Tests for the MIST-Transformer: forecast contract and the no-leakage guarantee.

The leakage test is the important one: because MIST assembles its spatial panel
from the store at inference time, we must prove it only ever uses data issued on
or before the forecast date. We do this by forecasting, then injecting absurd
*future-issued* data into the same store and confirming the forecast is byte-for-
byte unchanged.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402
from models.mist_transformer import MISTModel, estimate_rt  # noqa: E402
import torch  # noqa: E402

LAG = pd.Timedelta(days=4)
LOCS = ["US", "06", "36"]


def _panel_store(tmp_path, n_weeks=36):
    """Synthetic 3-location weekly panel; week issued 4 days after reference date."""
    dates = pd.date_range("2023-09-30", periods=n_weeks, freq="7D")
    rng = np.random.default_rng(3)
    rows = []
    for k, loc in enumerate(LOCS):
        t = np.linspace(0, np.pi, n_weeks)
        base = (200 + k * 50) + (1500 + 300 * k) * np.sin(t) ** 2
        vals = np.clip(base + rng.normal(0, 20, n_weeks), 0, None)
        for d, v in zip(dates, vals):
            rows.append(("synthetic", "flu_hosp_admissions", loc, d, d + LAG, float(v)))
    s = VersionedStore(store_dir=tmp_path / "store")
    s.ingest(pd.DataFrame(rows, columns=SCHEMA_COLUMNS))
    return s, dates


def _trained_model(store, train_end):
    model = MISTModel(context_length=8, horizon=2, patch_sizes=(2, 4),
                      d_model=16, n_heads=2, epochs=5, batch_size=8, seed=0)
    model.fit(store, signal="flu_hosp_admissions", source="synthetic",
              locations=LOCS, train_end_date=train_end, verbose=False)
    return model


def test_rt_tracks_growth_and_decline():
    rise = torch.tensor([[[10, 12, 15, 18, 22, 27, 33, 40.]]])
    fall = torch.tensor([[[40, 33, 27, 22, 18, 15, 12, 10.]]])
    assert estimate_rt(rise)[0, 0, -1] > 1.0
    assert estimate_rt(fall)[0, 0, -1] < 1.0


def test_forecast_contract(tmp_path):
    store, dates = _panel_store(tmp_path)
    model = _trained_model(store, train_end=dates[20])
    assert model.W.shape == (len(LOCS), len(LOCS))

    fd = dates[28] + LAG
    hist = store.get_vintage("flu_hosp_admissions", "06", fd, source="synthetic")
    preds = model.predict(hist, fd, horizons=(1, 2))
    assert set(preds["horizon"]) == {1, 2}
    for h, g in preds.groupby("horizon"):
        v = g.sort_values("quantile")["value"].to_numpy()
        assert len(v) == 23
        assert np.all(v >= 0)
        assert np.all(np.diff(v) >= -1e-6)


def test_no_leakage_future_data_does_not_change_forecast(tmp_path):
    """Injecting future-issued data must not change a forecast made earlier."""
    store, dates = _panel_store(tmp_path)
    model = _trained_model(store, train_end=dates[20])

    fd = dates[26] + LAG
    hist = store.get_vintage("flu_hosp_admissions", "US", fd, source="synthetic")
    model._cache.clear()
    before = model.predict(hist, fd, horizons=(1, 2))["value"].to_numpy()

    # Inject absurd values issued AFTER fd (and for future reference dates).
    future_rows = []
    for k in range(1, 6):
        ref = fd + pd.Timedelta(weeks=k)
        for loc in LOCS:
            future_rows.append(("synthetic", "flu_hosp_admissions", loc,
                                ref, fd + pd.Timedelta(days=90), 999999.0))
    store.ingest(pd.DataFrame(future_rows, columns=SCHEMA_COLUMNS))

    model._cache.clear()
    after = model.predict(hist, fd, horizons=(1, 2))["value"].to_numpy()

    assert np.allclose(before, after), "future-issued data leaked into the forecast"

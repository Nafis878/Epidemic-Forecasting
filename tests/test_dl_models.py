"""Fast smoke tests for the deep-learning baselines (tiny training on synthetic).

Verifies the train -> predict path and the 23-quantile forecast contract the
back-tester depends on. Kept tiny (few epochs, synthetic data) so the suite
stays quick.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402


def _synthetic_series(n_series=8, length=60):
    rng = np.random.default_rng(0)
    series = []
    for _ in range(n_series):
        t = np.linspace(0, 3 * np.pi, length)
        s = 100 + 50 * np.sin(t + rng.uniform(0, 1)) + rng.normal(0, 3, length)
        series.append(np.clip(s, 0, None).astype(np.float32))
    return series


def _history(length=20):
    dates = pd.date_range("2024-01-06", periods=length, freq="7D")
    t = np.linspace(0, np.pi, length)
    return pd.DataFrame({"reference_date": dates,
                         "value": 100 + 80 * np.sin(t), "location": "US"})


@pytest.mark.parametrize("ModelCls", [TFTModel, PatchTSTModel])
def test_train_then_predict_contract(ModelCls):
    model = ModelCls(context_length=12, horizon=4, epochs=3, batch_size=32, seed=0)
    model.fit_series(_synthetic_series())

    hist = _history()
    last_date = hist["reference_date"].max()
    preds = model.predict(hist, last_date, horizons=(1, 2, 3, 4))

    assert set(preds["horizon"]) == {1, 2, 3, 4}
    for h, g in preds.groupby("horizon"):
        g = g.sort_values("quantile")
        v = g["value"].to_numpy()
        assert len(v) == 23
        assert np.all(v >= 0)                  # non-negative counts
        assert np.all(np.diff(v) >= -1e-6)     # monotone quantiles (sorted in predict)
        assert pd.Timestamp(g["reference_date"].iloc[0]) == last_date + pd.Timedelta(days=7 * h)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TFTModel().predict(_history(), "2024-05-01", horizons=(1,))

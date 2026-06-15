"""Leakage guard for the phase-stacked hybrid.

The whole hybrid hinges on one property: the Hedge weights used to combine base
models at origin ``t`` may depend ONLY on forecasts whose target week was
realised strictly before ``t``. These tests construct synthetic loss histories
and prove that future (>= t) information cannot influence the weights at ``t``.
"""

import numpy as np
import pandas as pd

from models.phase_stack import _origin_weights

COMPONENTS = ["A", "B"]
T = pd.Timestamp("2024-01-01")


def _wis_row(model, fdate, rdate, wis):
    return {"model": model, "season": "s", "forecast_date": pd.Timestamp(fdate),
            "location": "01", "horizon": 1, "reference_date": pd.Timestamp(rdate),
            "y_true": 1.0, "phase_origin": "Rising", "wis": float(wis)}


def _history(future_wis_A=99.0, future_wis_B=0.1):
    """40 prior forecasts where A is better; 40 future forecasts where B is better."""
    rows = []
    # Prior: realised strictly before T (A low loss, B high loss).
    for i in range(40):
        rd = T - pd.Timedelta(days=7 * (i + 1))
        fd = rd - pd.Timedelta(days=7)
        rows.append(_wis_row("A", fd, rd, 1.0))
        rows.append(_wis_row("B", fd, rd, 50.0))
    # Future: realised at/after T (deliberately the opposite ordering). An origin
    # at T must NOT see these.
    for i in range(40):
        rd = T + pd.Timedelta(days=7 * i)
        rows.append(_wis_row("A", T, rd, future_wis_A))
        rows.append(_wis_row("B", T, rd, future_wis_B))
    return pd.DataFrame(rows)


def test_weights_at_t_use_only_pre_t_losses():
    w = _origin_weights(_history(), COMPONENTS, eta=0.1, by_phase=False)
    key = (1, T)                                   # (horizon, origin)
    assert key in w
    wa, wb = w[key]
    assert wa > wb                                 # A won the prior pool -> heavier


def test_future_losses_cannot_change_weights_at_t():
    w_default = _origin_weights(_history(), COMPONENTS, eta=0.1, by_phase=False)[(1, T)]
    # Corrupt the FUTURE losses drastically (flip who looks good after T).
    w_corrupt = _origin_weights(_history(future_wis_A=0.001, future_wis_B=999.0),
                                COMPONENTS, eta=0.1, by_phase=False)[(1, T)]
    assert np.allclose(w_default, w_corrupt), "future forecasts leaked into origin-t weights"


def test_thin_history_falls_back_to_equal_weight():
    rows = [_wis_row("A", "2023-12-01", "2023-12-08", 1.0),
            _wis_row("B", "2023-12-01", "2023-12-08", 50.0)]  # only 2 prior < T
    w = _origin_weights(pd.DataFrame(rows), COMPONENTS, eta=0.1, by_phase=False)
    # Any origin after this single realised week has < MIN_HISTORY prior -> equal.
    for (h, t), vec in w.items():
        assert np.allclose(vec, [0.5, 0.5])

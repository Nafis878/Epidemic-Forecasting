"""Phase 4 tests: Diebold-Mariano test behaviour and spatial connectivity strata."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dm_test import dm_test  # noqa: E402
from evaluation.spatial import connectivity_strata  # noqa: E402
from features.mobility import gravity_matrix, total_outflow  # noqa: E402


def _frame(wis, model="m"):
    n = len(wis)
    return pd.DataFrame({"location": ["US"] * n,
                         "forecast_date": pd.date_range("2024-01-06", periods=n, freq="7D"),
                         "horizon": 1, "wis": wis, "model": model})


def test_dm_identical_models_not_significant():
    rng = np.random.default_rng(0)
    w = rng.uniform(10, 100, 40)
    r = dm_test(_frame(w), _frame(w.copy()))
    assert abs(r["mean_diff"]) < 1e-9
    assert np.isnan(r["p"]) or r["p"] > 0.5


def test_dm_detects_clear_winner():
    rng = np.random.default_rng(1)
    base = rng.uniform(50, 60, 60)
    a = base                              # model A clearly lower loss
    b = base + 40.0                       # model B consistently worse
    r = dm_test(_frame(a), _frame(b))
    assert r["mean_diff"] < 0             # A better
    assert r["p"] < 0.05


def test_gravity_matrix_symmetric_unit_diagonal():
    locs = ["US", "06", "36", "48"]
    W = gravity_matrix(locs)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 1.0)
    assert (W >= 0).all() and (W <= 1.0 + 1e-9).all()


def test_connectivity_strata_assigns_tiers():
    locs = ["US", "06", "36", "48", "56", "50", "72"]   # incl CA/TX hubs, WY/VT small, PR terr
    strata = connectivity_strata(locs).set_index("location")
    assert strata.loc["72", "tier"] == "territory"
    assert strata.loc["US", "tier"] == "national"
    # A big state should out-rank a tiny one in outflow.
    out = total_outflow(locs)
    assert out["06"] > out["56"]

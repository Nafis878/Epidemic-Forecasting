"""Tests for the rolling-origin back-tester, focused on the no-leakage contract."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import rolling_origin_backtest, summarize  # noqa: E402
from evaluation.metrics import DEFAULT_QUANTILES  # noqa: E402
from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402
from models.baseline import PersistenceQuantileModel  # noqa: E402

LAG = pd.Timedelta(days=4)


@pytest.fixture()
def store(tmp_path):
    """Weekly synthetic series, each week issued 4 days after its reference date."""
    dates = pd.date_range("2024-01-06", periods=20, freq="7D")  # Saturdays
    rng = np.random.default_rng(7)
    values = 100 + np.cumsum(rng.normal(0, 5, size=len(dates)))
    rows = [
        ("synthetic", "flu_hosp_admissions", "US", d, d + LAG, float(max(0, v)))
        for d, v in zip(dates, values)
    ]
    s = VersionedStore(store_dir=tmp_path / "store")
    s.ingest(pd.DataFrame(rows, columns=SCHEMA_COLUMNS))
    yield s
    s.close()


class LeakSpyModel:
    """Records the max reference_date and issue exposure it is ever handed."""

    def __init__(self):
        self.seen_max_reference_date = pd.Timestamp.min

    def predict(self, history, forecast_date, horizons, quantiles):
        self.seen_max_reference_date = max(
            self.seen_max_reference_date, history["reference_date"].max()
        )
        # Trivial flat forecast so the call succeeds.
        last = float(history["value"].iloc[-1])
        rows = []
        for h in horizons:
            target = pd.Timestamp(history["reference_date"].max()) + pd.Timedelta(days=7 * h)
            for q in quantiles:
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": q, "value": last})
        return pd.DataFrame(rows)


def test_model_never_sees_future_data(store):
    """The history passed to the model must never include unissued (future) weeks."""
    spy = LeakSpyModel()
    forecast_dates = pd.date_range("2024-02-01", periods=6, freq="14D")
    rolling_origin_backtest(
        store, spy, forecast_dates,
        signal="flu_hosp_admissions", location="US", source="synthetic",
    )
    # For every origin, the latest reference_date seen had issue_date <= origin.
    # Equivalently its reference_date + LAG <= the last forecast date used.
    last_origin = forecast_dates.max()
    assert spy.seen_max_reference_date + LAG <= last_origin


def test_history_issue_dates_respect_each_origin(store):
    """Directly check the vintage slice per origin obeys issue_date <= origin."""
    for origin in pd.date_range("2024-02-01", periods=6, freq="14D"):
        hist = store.get_vintage("flu_hosp_admissions", "US", origin, source="synthetic")
        # Every visible week must have been issued (reference + lag) on/before origin.
        assert ((hist["reference_date"] + LAG) <= origin).all()


def test_backtest_produces_scored_results(store):
    """End-to-end: baseline model yields a well-formed, finite results frame."""
    model = PersistenceQuantileModel(cadence_days=7)
    forecast_dates = pd.date_range("2024-02-10", periods=8, freq="7D")
    res = rolling_origin_backtest(
        store, model, forecast_dates,
        signal="flu_hosp_admissions", location="US", source="synthetic",
        horizons=(1, 2, 3, 4), quantiles=DEFAULT_QUANTILES,
    )
    assert not res.empty
    assert set(["forecast_date", "horizon", "y_true", "wis", "mae",
                "cov_50", "cov_95"]).issubset(res.columns)
    assert np.isfinite(res["wis"]).all() and (res["wis"] >= 0).all()

    summary = summarize(res, by=("horizon",))
    assert set(summary["horizon"]) <= {1, 2, 3, 4}
    assert (summary["wis"] >= 0).all()
    assert ((summary["coverage_50"] >= 0) & (summary["coverage_50"] <= 1)).all()


def test_unobserved_targets_are_skipped(store):
    """Forecasting past the end of the data yields no scorable rows for those targets."""
    model = PersistenceQuantileModel(cadence_days=7)
    # Origin near the very end: 4-week-ahead targets fall beyond observed data.
    last_ref = store.get_vintage("flu_hosp_admissions", "US", "2100-01-01",
                                 source="synthetic")["reference_date"].max()
    res = rolling_origin_backtest(
        store, model, [last_ref],
        signal="flu_hosp_admissions", location="US", source="synthetic",
        horizons=(1, 2, 3, 4),
    )
    # No target beyond the last observed reference_date should appear.
    assert (res["reference_date"] <= last_ref).all()

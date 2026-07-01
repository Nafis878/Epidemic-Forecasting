"""WP0: revision-triangle accessor for the revision-aware forecaster (RAF).

Validates that ``VersionedStore.get_triangle`` exposes the full backfill history
knowable at an origin *without leakage* (no ``issue_date > forecast_date``), and
that ``to_lag_matrix`` reshapes it into the dense ``(reference_week x issue_lag)``
form RAF consumes — with NaNs exactly on the not-yet-issued cells.
"""

import numpy as np
import pandas as pd

from features.versioned_store import VersionedStore


def _revised_payload():
    """Two reference weeks, each revised upward over successive weekly issues."""
    return pd.DataFrame([
        # week ending 2023-10-07: issued at lag 0, 1, 2 (100 -> 120 -> 130)
        {"source": "nhsn", "signal": "flu", "location": "CA",
         "reference_date": "2023-10-07", "issue_date": "2023-10-07", "value": 100.0},
        {"source": "nhsn", "signal": "flu", "location": "CA",
         "reference_date": "2023-10-07", "issue_date": "2023-10-14", "value": 120.0},
        {"source": "nhsn", "signal": "flu", "location": "CA",
         "reference_date": "2023-10-07", "issue_date": "2023-10-21", "value": 130.0},
        # week ending 2023-10-14: issued at lag 0, 1 (80 -> 95)
        {"source": "nhsn", "signal": "flu", "location": "CA",
         "reference_date": "2023-10-14", "issue_date": "2023-10-14", "value": 80.0},
        {"source": "nhsn", "signal": "flu", "location": "CA",
         "reference_date": "2023-10-14", "issue_date": "2023-10-21", "value": 95.0},
    ])


def _store(tmp_path):
    store = VersionedStore(store_dir=str(tmp_path / "store"))
    store.ingest(_revised_payload())
    return store


def test_triangle_returns_all_revisions_and_lag(tmp_path):
    store = _store(tmp_path)
    tri = store.get_triangle("flu", "CA", "2023-10-21", source="nhsn")
    assert len(tri) == 5                                   # every revision present
    assert "issue_lag_weeks" in tri.columns
    wk40 = tri[tri["reference_date"] == pd.Timestamp("2023-10-07")]
    assert sorted(wk40["issue_lag_weeks"]) == [0, 1, 2]
    assert sorted(wk40["value"]) == [100.0, 120.0, 130.0]


def test_triangle_is_leakage_safe(tmp_path):
    store = _store(tmp_path)
    # As of 2023-10-14 the lag-2 revision of week 40 (issued 10-21) is unknowable.
    tri = store.get_triangle("flu", "CA", "2023-10-14", source="nhsn")
    assert (tri["issue_date"] <= pd.Timestamp("2023-10-14")).all()
    wk40 = tri[tri["reference_date"] == pd.Timestamp("2023-10-07")]
    assert sorted(wk40["issue_lag_weeks"]) == [0, 1]       # lag 2 not yet issued
    assert 130.0 not in set(wk40["value"])                 # the future revision is hidden


def test_lag_matrix_shape_and_nan_frontier(tmp_path):
    store = _store(tmp_path)
    tri = store.get_triangle("flu", "CA", "2023-10-14", source="nhsn")
    weeks, mat = VersionedStore.to_lag_matrix(tri, max_lag=2)
    assert mat.shape == (2, 3)                              # 2 weeks x lags {0,1,2}
    # week 40 as-of 10-14: lag0=100, lag1=120, lag2 not-yet-issued -> NaN.
    row40 = mat[list(weeks).index(pd.Timestamp("2023-10-07"))]
    assert row40[0] == 100.0 and row40[1] == 120.0 and np.isnan(row40[2])
    # week 41 as-of 10-14: only lag0=80 known; lag1,2 are NaN.
    row41 = mat[list(weeks).index(pd.Timestamp("2023-10-14"))]
    assert row41[0] == 80.0 and np.isnan(row41[1]) and np.isnan(row41[2])


def test_lag_matrix_last_n_trims_oldest(tmp_path):
    store = _store(tmp_path)
    tri = store.get_triangle("flu", "CA", "2023-10-21", source="nhsn")
    weeks, mat = VersionedStore.to_lag_matrix(tri, max_lag=2, last_n=1)
    assert len(weeks) == 1 and weeks[0] == pd.Timestamp("2023-10-14")  # newest week only


def test_triangle_empty_when_no_data(tmp_path):
    store = VersionedStore(store_dir=str(tmp_path / "empty"))
    tri = store.get_triangle("flu", "CA", "2023-10-21", source="nhsn")
    assert tri.empty and "issue_lag_weeks" in tri.columns
    weeks, mat = VersionedStore.to_lag_matrix(tri, max_lag=2)
    assert len(weeks) == 0 and mat.shape == (0, 3)

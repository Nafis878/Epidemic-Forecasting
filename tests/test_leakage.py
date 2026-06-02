"""Anti-leakage tests for :class:`features.versioned_store.VersionedStore`.

The contract under test: ``get_vintage(signal, location, forecast_date)`` must
behave exactly as a forecaster standing on ``forecast_date`` would experience the
data — it must never reveal any value that was issued *after* that date, and when
a ``reference_date`` has been revised it must return the revision in force on the
forecast date, not a later (future) correction.
"""

import os
import sys

import pandas as pd
import pytest

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    """A store seeded with a revised signal across several issue dates.

    Scenario for signal=flu_admissions / location=US:
      - reference_date 2024-01-06 first issued 2024-01-09 (value 100),
        then revised 2024-01-16 (value 110) and again 2024-02-01 (value 115).
      - reference_date 2024-01-13 issued 2024-01-16 (value 200).
      - reference_date 2024-01-20 issued 2024-01-23 (value 300)  <- "future".
    """
    s = VersionedStore(store_dir=tmp_path / "store")
    rows = [
        ("nhsn", "flu_admissions", "US", "2024-01-06", "2024-01-09", 100.0),
        ("nhsn", "flu_admissions", "US", "2024-01-06", "2024-01-16", 110.0),
        ("nhsn", "flu_admissions", "US", "2024-01-06", "2024-02-01", 115.0),
        ("nhsn", "flu_admissions", "US", "2024-01-13", "2024-01-16", 200.0),
        ("nhsn", "flu_admissions", "US", "2024-01-20", "2024-01-23", 300.0),
        # An unrelated series that must never bleed into US results.
        ("nhsn", "flu_admissions", "CA", "2024-01-06", "2024-01-09", 42.0),
    ]
    s.ingest(pd.DataFrame(rows, columns=SCHEMA_COLUMNS))
    yield s
    s.close()


def test_no_future_issue_dates_are_visible(store):
    """Core guarantee: nothing issued after the forecast date is returned."""
    forecast_date = pd.Timestamp("2024-01-17")
    vintage = store.get_vintage("flu_admissions", "US", forecast_date, latest_only=False)
    assert not vintage.empty
    assert (vintage["issue_date"] <= forecast_date).all(), (
        "get_vintage leaked rows issued after the forecast date"
    )


def test_revision_in_force_is_returned_not_future_correction(store):
    """As of 2024-01-17 the 2024-01-06 value is the 110 revision, not 100 or 115."""
    vintage = store.get_vintage("flu_admissions", "US", "2024-01-17")  # latest_only=True
    ref = vintage[vintage["reference_date"] == pd.Timestamp("2024-01-06")]
    assert len(ref) == 1, "vintage view must return exactly one row per reference_date"
    assert ref["value"].iloc[0] == 110.0


def test_earliest_vintage_sees_only_first_release(store):
    """As of 2024-01-10 only the very first release (value 100) exists."""
    vintage = store.get_vintage("flu_admissions", "US", "2024-01-10")
    assert list(vintage["reference_date"].dt.strftime("%Y-%m-%d")) == ["2024-01-06"]
    assert vintage["value"].iloc[0] == 100.0


def test_future_reference_date_excluded_until_issued(store):
    """The 2024-01-20 observation (issued 2024-01-23) is invisible on 2024-01-17."""
    vintage = store.get_vintage("flu_admissions", "US", "2024-01-17")
    assert pd.Timestamp("2024-01-20") not in set(vintage["reference_date"])
    # ...but it appears once the forecast date reaches its issue date.
    later = store.get_vintage("flu_admissions", "US", "2024-01-25")
    assert pd.Timestamp("2024-01-20") in set(later["reference_date"])


def test_final_vintage_sees_latest_revision(store):
    """Far in the future, the 2024-01-06 value settles on the 115 revision."""
    vintage = store.get_vintage("flu_admissions", "US", "2024-12-31")
    ref = vintage[vintage["reference_date"] == pd.Timestamp("2024-01-06")]
    assert ref["value"].iloc[0] == 115.0


def test_location_filter_isolated(store):
    """A query for US must not return CA rows (and vice versa)."""
    us = store.get_vintage("flu_admissions", "US", "2024-12-31")
    assert set(us["location"]) == {"US"}
    ca = store.get_vintage("flu_admissions", "CA", "2024-12-31")
    assert set(ca["location"]) == {"CA"}


def test_empty_store_returns_empty_frame(tmp_path):
    """Querying an empty store yields an empty, correctly-shaped frame."""
    s = VersionedStore(store_dir=tmp_path / "empty")
    out = s.get_vintage("flu_admissions", "US", "2024-01-01")
    assert out.empty
    assert list(out.columns) == SCHEMA_COLUMNS
    s.close()


def test_missing_columns_rejected(tmp_path):
    """Ingesting a frame without the required schema raises."""
    s = VersionedStore(store_dir=tmp_path / "bad")
    with pytest.raises(ValueError):
        s.ingest(pd.DataFrame({"signal": ["x"], "value": [1.0]}))
    s.close()

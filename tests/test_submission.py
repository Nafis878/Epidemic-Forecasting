"""WP4: the external-submission scorer (forecast-hub-style leaderboard entry)."""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import DEFAULT_QUANTILES, weighted_interval_score
from evaluation.run_seasons import SIG, SRC
from features.versioned_store import VersionedStore
from scripts.evaluate_submission import score


def _store_with_truth(tmp_path):
    store = VersionedStore(store_dir=str(tmp_path / "store"))
    rows = []
    for rd, val in [("2023-11-11", 500.0), ("2023-11-18", 600.0)]:
        rows.append({"source": SRC, "signal": SIG, "location": "06",
                     "reference_date": rd, "issue_date": "2023-11-25", "value": val})
    store.ingest(pd.DataFrame(rows))
    return store


def _submission(values):
    rows = []
    for fd, rd in [("2023-11-04", "2023-11-11"), ("2023-11-11", "2023-11-18")]:
        for q in DEFAULT_QUANTILES:
            rows.append({"forecast_date": pd.Timestamp(fd), "location": "06", "horizon": 1,
                         "reference_date": pd.Timestamp(rd), "quantile": q,
                         "value": np.quantile(values, q)})
    return pd.DataFrame(rows)


def test_score_matches_direct_wis_and_infers_season(tmp_path):
    store = _store_with_truth(tmp_path)
    sub = _submission(np.linspace(300, 700, 50))
    scored = score(sub, store=store)
    assert len(scored) == 2
    assert set(scored["season"]) == {"2023-24"}            # inferred from forecast_date
    g = sub[sub["reference_date"] == pd.Timestamp("2023-11-11")].sort_values("quantile")
    direct = weighted_interval_score(500.0, g["value"].to_numpy(), g["quantile"].to_numpy())
    got = scored[scored["reference_date"] == pd.Timestamp("2023-11-11")]["wis"].iloc[0]
    assert np.isclose(got, direct, rtol=1e-6)


def test_unrealised_targets_are_skipped(tmp_path):
    store = _store_with_truth(tmp_path)
    sub = _submission(np.linspace(300, 700, 50))
    # add a forecast whose target has no truth in the store -> must be skipped
    extra = sub.copy()
    extra["forecast_date"] = pd.Timestamp("2024-11-02")
    extra["reference_date"] = pd.Timestamp("2024-11-09")
    scored = score(pd.concat([sub, extra], ignore_index=True), store=store)
    assert len(scored) == 2                                 # extra dropped


def test_missing_columns_raises(tmp_path):
    store = _store_with_truth(tmp_path)
    bad = _submission(np.linspace(1, 2, 5)).drop(columns=["quantile"])
    with pytest.raises(ValueError):
        score(bad, store=store)

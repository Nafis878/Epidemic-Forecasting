"""Multi-season assembly + aggregation correctness.

Integration tests over the persisted quantile dump. They are skipped when
``results/quantiles_long.parquet`` has not been generated yet (run
``python scripts/reproduce.py --quick`` first); CI's reproduce step creates it.
"""

import os

import numpy as np
import pytest

from evaluation.ensembles import ENSEMBLE_NAMES, LONG_PATH, per_forecast_metrics
from evaluation.metrics import weighted_interval_score

pytestmark = pytest.mark.skipif(
    not os.path.exists(LONG_PATH),
    reason="quantiles_long.parquet not generated (run scripts/reproduce.py --quick)",
)

_STACKS = {"stack_global", "stack_phase", "stack_phase_conformal"}


@pytest.fixture(scope="module")
def assembled():
    from evaluation.run_multiseason import assemble_all
    return assemble_all()


def test_assembly_contains_base_ensembles_and_hybrids(assembled):
    models = set(assembled["model"].unique())
    assert _STACKS <= models
    assert set(ENSEMBLE_NAMES) <= models
    assert {"mist_v2", "patchtst", "tft", "arima", "persistence"} <= models


def test_per_forecast_wis_matches_direct_metric(assembled):
    metrics = per_forecast_metrics(assembled)
    keycols = ["season", "model", "forecast_date", "location", "horizon", "reference_date"]
    # Pick one forecast's quantile vector via boolean masking (avoids dtype-mixed merges).
    first = assembled.iloc[0]
    mask = np.logical_and.reduce([assembled[c] == first[c] for c in keycols])
    g = assembled[mask].sort_values("quantile")
    direct = weighted_interval_score(g["y_true"].iloc[0], g["value"].to_numpy(),
                                     g["quantile"].to_numpy())
    mmask = np.logical_and.reduce([metrics[c] == first[c] for c in keycols])
    assert np.isclose(metrics[mmask]["wis"].iloc[0], direct, rtol=1e-6)


def test_season_unweighted_is_mean_of_per_season_means(assembled):
    from evaluation.run_multiseason import _multiseason_summary, _season_leaderboard
    metrics = per_forecast_metrics(assembled)
    lb = _season_leaderboard(metrics)
    summary = _multiseason_summary(lb, metrics).set_index("model")
    m = "ens_perf"
    by_season = lb[lb["model"] == m]["wis"]
    assert np.isclose(summary.loc[m, "wis_unweighted"], by_season.mean(), rtol=1e-6)


def test_hybrid_quantiles_are_monotone(assembled):
    hy = assembled[assembled["model"] == "stack_phase_conformal"]
    for _, g in hy.groupby(["season", "forecast_date", "location", "horizon"]):
        v = g.sort_values("quantile")["value"].to_numpy()
        assert np.all(np.diff(v) >= -1e-6)

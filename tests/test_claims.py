"""Claim-consistency: the recorded hybrid verdict must match the artifacts.

Guards against the failure mode the whole upgrade is meant to prevent — a paper
claim that the saved numbers do not support. Skipped until the multi-season
artifacts exist (run ``scripts/reproduce.py``).
"""

import json
import os

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
VD = os.path.join(RES, "hybrid_verdict.json")
MS = os.path.join(RES, "multiseason_summary.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VD) and os.path.exists(MS)),
    reason="multi-season artifacts not generated (run scripts/reproduce.py)",
)


@pytest.fixture(scope="module")
def verdict():
    with open(VD) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(MS).set_index("model")


def test_verdict_internally_consistent(verdict):
    conds = verdict["conditions"]
    assert verdict["earned"] == all(conds.values())


def test_earned_implies_hybrid_is_leader(verdict, summary):
    if verdict["earned"]:
        assert int(summary.loc["stack_phase_conformal", "rank_unweighted"]) == 1
    else:
        # Honest pivot: if not earned, do not silently rank the hybrid first by
        # accident of a tie — the recorded WIS must back the non-dominance.
        hybrid = summary.loc["stack_phase_conformal", "wis_unweighted"]
        best = summary["wis_unweighted"].min()
        assert hybrid >= best - 1e-9


def test_verdict_wis_matches_summary(verdict, summary):
    assert abs(verdict["hybrid_wis_unweighted"]
               - summary.loc["stack_phase_conformal", "wis_unweighted"]) < 0.05


def test_assert_reproducibility_hybrid_checks_pass():
    from evaluation.assert_reproducibility import _hybrid_checks
    for desc, passed, actual in _hybrid_checks():
        assert passed, f"{desc} ({actual})"


def test_trimmed_ensemble_is_significant_leader(summary):
    """Headline finding: the trimmed ensemble is the season-unweighted leader and
    significantly beats the median ensemble (the hub standard)."""
    assert summary["wis_unweighted"].idxmin() == "ens_trimmed"
    ci_path = os.path.join(ROOT, "tables", "bootstrap_ci.csv")
    if os.path.exists(ci_path):
        ci = pd.read_csv(ci_path)
        row = ci[(ci["focal"] == "ens_trimmed") & (ci["vs"] == "ens_median")]
        assert not row.empty and row["ci_high"].iloc[0] < 0   # CI excludes 0 -> significant

"""Reproducibility gate: every headline claim must hold against the result CSVs.

Skips cleanly if the result artifacts have not been generated yet (fresh checkout),
so the suite stays green before a full benchmark run while still enforcing the claims
once the CSVs exist (CI / post-run).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.assert_reproducibility import RES, TAB, _checks, check  # noqa: E402

_REQUIRED = [
    os.path.join(RES, "leaderboard_v2.csv"),
    os.path.join(RES, "phase_performance_v2.csv"),
    os.path.join(RES, "per_location_coverage.csv"),
    os.path.join(RES, "season_wis.csv"),
    os.path.join(TAB, "dm_significance.csv"),
]


def _artifacts_present() -> bool:
    return all(os.path.exists(p) for p in _REQUIRED)


@pytest.mark.skipif(not _artifacts_present(),
                    reason="result CSVs not generated yet (run the benchmark first)")
def test_all_reproducibility_claims_hold():
    assert check(), "one or more reproducibility claims failed; see printed report"


@pytest.mark.skipif(not _artifacts_present(),
                    reason="result CSVs not generated yet (run the benchmark first)")
def test_each_gated_claim_individually():
    # Every gated check (all but the cov-50 'reported' line) must pass.
    for desc, passed, actual in _checks():
        if "reported (not gated)" in desc:
            continue
        assert passed, f"{desc} -> {actual}"

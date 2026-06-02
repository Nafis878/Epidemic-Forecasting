"""Phase 1 tests: crosswalk completeness, NHSN reporting-shift flag, ILINet vintaging.

These cover the data-integrity work (Tasks 1.1-1.3) without requiring network
access — the ILINet test exercises the MMWR date conversion and the schema
mapping on a small synthetic frame.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.crosswalk import (build_crosswalk, fips_to_ilinet,  # noqa: E402
                                 validate)
from ingestion.ilinet import (date_to_epiweek, epiweek_range,  # noqa: E402
                              epiweek_to_saturday, to_schema)
from ingestion.nhsn import NHSN_MANDATE_DATE, post_mandatory_flag  # noqa: E402


# ------------------------------------------------------------------ crosswalk 1.3
def test_crosswalk_has_all_flusight_states():
    cw = build_crosswalk()
    # 50 states + DC + national + PR are the FluSight locations we must cover.
    fips = set(cw["fips"])
    for f in ["US", "06", "36", "11", "72"]:
        assert f in fips
    assert cw["fips"].is_unique and cw["abbrev_upper"].is_unique


def test_crosswalk_national_and_pr_mapping():
    assert fips_to_ilinet("US") == "nat"
    assert fips_to_ilinet("06") == "ca"
    # Puerto Rico has no ILINet series -> None (documented gap).
    assert fips_to_ilinet("72") is None


def test_crosswalk_validate_flags_only_pr():
    cw = build_crosswalk()
    flusight = ["US", "06", "36", "72"]
    nhsn = ["USA", "CA", "NY", "PR"]
    ilinet = ["nat", "ca", "ny"]              # no 'pr'
    warns = validate(cw, flusight, nhsn, ilinet)
    assert len(warns) == 1 and "72" in warns[0]


# -------------------------------------------------------------------- NHSN 1.1
def test_post_mandatory_flag():
    dates = pd.to_datetime(["2024-10-26", "2024-11-01", "2025-01-04"])
    flags = post_mandatory_flag(dates).tolist()
    assert flags == [False, True, True]
    assert NHSN_MANDATE_DATE == pd.Timestamp("2024-11-01")


# ------------------------------------------------------------------- ILINet 1.2
def test_epiweek_saturday_roundtrip():
    for ew in [199740, 202001, 202240, 202252, 202301, 202340, 202401]:
        sat = epiweek_to_saturday(ew)
        assert sat.day_name() == "Saturday"
        assert date_to_epiweek(sat) == ew


def test_epiweek_range_inclusive_and_ordered():
    rng = epiweek_range(202240, 202301)
    assert rng[0] == 202240 and rng[-1] == 202301
    assert rng == sorted(rng)


def test_ilinet_to_schema_uses_real_issue_date_and_weighted_national():
    raw = pd.DataFrame({
        "region": ["nat", "ca"],
        "epiweek": [202301, 202301],
        "issue": [202305, 202302],          # issued AFTER the reference week
        "ili": [2.0, 3.0],
        "wili": [2.5, 9.9],                  # national uses wili, state uses ili
    })
    sch = to_schema(raw)
    nat = sch[sch["location"] == "nat"].iloc[0]
    ca = sch[sch["location"] == "ca"].iloc[0]
    assert nat["value"] == 2.5 and ca["value"] == 3.0
    # issue_date is the Saturday of the *issue* epiweek, strictly after reference.
    assert nat["issue_date"] == epiweek_to_saturday(202305)
    assert nat["issue_date"] > nat["reference_date"]
    assert nat["signal"] == "ili_pct" and nat["source"] == "ilinet"

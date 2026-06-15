"""WP1: genuine-vintage ingestion core + authenticity report.

Validates the pure, network-free parts: the MMWR epiweek->date helper, the
covidcast revision normaliser (which must preserve multiple issues per reference
week), and that such data registers as GENUINE in evaluation.vintage_report.
"""

from datetime import date

import pandas as pd

from evaluation.vintage_report import audit
from features.versioned_store import VersionedStore
from ingestion.vintage_delphi import epiweek_to_saturday, normalise_covidcast


def test_epiweek_to_saturday_known_values():
    assert epiweek_to_saturday(202401) == date(2024, 1, 6)
    assert epiweek_to_saturday(202340) == date(2023, 10, 7)
    assert epiweek_to_saturday(202101) == date(2021, 1, 9)


def _payload():
    # Same (geo, time_value) revised across two issues -> genuine vintage.
    return [
        {"geo_value": "ca", "time_value": 202340, "issue": 202340, "value": 100.0},
        {"geo_value": "ca", "time_value": 202340, "issue": 202342, "value": 130.0},
        {"geo_value": "ca", "time_value": 202341, "issue": 202341, "value": 90.0},
        {"geo_value": "ny", "time_value": 202340, "issue": 202340, "value": None},  # dropped
    ]


def test_normalise_preserves_multiple_issues():
    df = normalise_covidcast(_payload(), signal="flu_hosp_admissions", source="delphi_nhsn")
    assert list(df.columns) == ["source", "signal", "location", "reference_date",
                                "issue_date", "value"]
    ca_wk40 = df[(df["location"] == "CA") &
                 (df["reference_date"] == pd.Timestamp("2023-10-07"))]
    assert len(ca_wk40) == 2                         # two genuine revisions kept
    assert set(ca_wk40["issue_date"]) == {pd.Timestamp("2023-10-07"),
                                          pd.Timestamp("2023-10-21")}
    assert (df["value"].isna().sum() == 0) and len(df) == 3  # None row dropped


def test_genuine_vintage_registers_as_authentic(tmp_path):
    df = normalise_covidcast(_payload(), signal="flu_hosp_admissions", source="delphi_nhsn")
    store = VersionedStore(store_dir=str(tmp_path / "store"))
    store.ingest(df)
    rep = audit(str(tmp_path / "store"))
    row = rep.iloc[0]
    assert bool(row["genuine_vintage"]) is True
    assert row["mean_issues_per_ref"] > 1.0
    # get_vintage as-of the first issue must NOT see the later revision.
    early = store.get_vintage("flu_hosp_admissions", "CA", "2023-10-10", source="delphi_nhsn")
    val = float(early[early["reference_date"] == pd.Timestamp("2023-10-07")]["value"].iloc[0])
    assert val == 100.0                              # pre-revision value, no leakage

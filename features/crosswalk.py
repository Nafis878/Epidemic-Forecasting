"""Location crosswalk across the three flu datasets.

The three sources code locations differently:

* **FluSight** — 2-digit FIPS (``US`` national, ``06`` California, ``72`` Puerto Rico).
* **NHSN HRD** — 2-letter jurisdiction abbreviations (``USA`` national; territories
  ``AS/GU/MP/VI/PR``; plus HHS ``Region 1..10`` rollups).
* **ILINet** (Delphi ``fluview``) — lowercase abbreviations (``nat`` national).

This module is the single source of truth that maps between them. It builds
``data/crosswalk.csv`` with columns::

    fips, abbrev_upper, abbrev_lower, state_name, nhsn_has_territory

and validates that every FluSight location maps to exactly one NHSN and one
ILINet location, printing any unmapped locations as warnings (notably Puerto
Rico, which NHSN reports but ILINet does not). The file is released in the repo
as a community contribution.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# fips -> (USPS abbrev, full name).  National + 50 states + DC + the 5 territories.
# Territory FIPS: PR 72, AS 60, GU 66, MP 69, VI 78.
_TABLE: list[tuple[str, str, str, bool]] = [
    # fips, abbrev_upper, state_name, is_territory
    ("US", "USA", "United States", False),
    ("01", "AL", "Alabama", False),
    ("02", "AK", "Alaska", False),
    ("04", "AZ", "Arizona", False),
    ("05", "AR", "Arkansas", False),
    ("06", "CA", "California", False),
    ("08", "CO", "Colorado", False),
    ("09", "CT", "Connecticut", False),
    ("10", "DE", "Delaware", False),
    ("11", "DC", "District of Columbia", False),
    ("12", "FL", "Florida", False),
    ("13", "GA", "Georgia", False),
    ("15", "HI", "Hawaii", False),
    ("16", "ID", "Idaho", False),
    ("17", "IL", "Illinois", False),
    ("18", "IN", "Indiana", False),
    ("19", "IA", "Iowa", False),
    ("20", "KS", "Kansas", False),
    ("21", "KY", "Kentucky", False),
    ("22", "LA", "Louisiana", False),
    ("23", "ME", "Maine", False),
    ("24", "MD", "Maryland", False),
    ("25", "MA", "Massachusetts", False),
    ("26", "MI", "Michigan", False),
    ("27", "MN", "Minnesota", False),
    ("28", "MS", "Mississippi", False),
    ("29", "MO", "Missouri", False),
    ("30", "MT", "Montana", False),
    ("31", "NE", "Nebraska", False),
    ("32", "NV", "Nevada", False),
    ("33", "NH", "New Hampshire", False),
    ("34", "NJ", "New Jersey", False),
    ("35", "NM", "New Mexico", False),
    ("36", "NY", "New York", False),
    ("37", "NC", "North Carolina", False),
    ("38", "ND", "North Dakota", False),
    ("39", "OH", "Ohio", False),
    ("40", "OK", "Oklahoma", False),
    ("41", "OR", "Oregon", False),
    ("42", "PA", "Pennsylvania", False),
    ("44", "RI", "Rhode Island", False),
    ("45", "SC", "South Carolina", False),
    ("46", "SD", "South Dakota", False),
    ("47", "TN", "Tennessee", False),
    ("48", "TX", "Texas", False),
    ("49", "UT", "Utah", False),
    ("50", "VT", "Vermont", False),
    ("51", "VA", "Virginia", False),
    ("53", "WA", "Washington", False),
    ("54", "WV", "West Virginia", False),
    ("55", "WI", "Wisconsin", False),
    ("56", "WY", "Wyoming", False),
    ("72", "PR", "Puerto Rico", True),
    ("60", "AS", "American Samoa", True),
    ("66", "GU", "Guam", True),
    ("69", "MP", "Northern Mariana Islands", True),
    ("78", "VI", "U.S. Virgin Islands", True),
]

# ILINet (fluview) carries the national series as ``nat`` and has no Puerto Rico
# or other-territory series. Everything else is just the lowercase abbreviation.
_ILINET_NATIONAL = "nat"
_ILINET_MISSING = {"PR", "AS", "GU", "MP", "VI"}  # not reported by ILINet


def build_crosswalk() -> pd.DataFrame:
    """Return the crosswalk DataFrame (does not write to disk)."""
    rows = []
    for fips, up, name, is_terr in _TABLE:
        if up == "USA":
            lower = _ILINET_NATIONAL
        elif up in _ILINET_MISSING:
            lower = None  # ILINet does not report this location
        else:
            lower = up.lower()
        rows.append({
            "fips": fips,
            "abbrev_upper": up,
            "abbrev_lower": lower,
            "state_name": name,
            "nhsn_has_territory": bool(is_terr),
        })
    return pd.DataFrame(rows)


def validate(cw: pd.DataFrame, flusight_locations, nhsn_locations,
             ilinet_regions) -> list[str]:
    """Check every FluSight location maps to exactly one NHSN and ILINet code.

    Returns a list of human-readable warning strings (also printed).
    """
    warnings: list[str] = []
    fl = set(map(str, flusight_locations))
    nhsn = set(map(str, nhsn_locations))
    ili = set(map(str, ilinet_regions))

    by_fips = cw.set_index("fips")
    for loc in sorted(fl):
        if loc not in by_fips.index:
            warnings.append(f"FluSight location {loc!r} is absent from the crosswalk.")
            continue
        row = by_fips.loc[loc]
        if row["abbrev_upper"] not in nhsn:
            warnings.append(
                f"FluSight {loc!r} ({row['state_name']}) -> NHSN "
                f"{row['abbrev_upper']!r} not present in NHSN data.")
        low = row["abbrev_lower"]
        if low is None or (isinstance(low, float) and pd.isna(low)) or low not in ili:
            warnings.append(
                f"FluSight {loc!r} ({row['state_name']}) has NO ILINet series "
                f"(expected {low!r}).")

    for w in warnings:
        print(f"  [crosswalk WARNING] {w}")
    if not warnings:
        print("  [crosswalk] all FluSight locations map to exactly one NHSN and ILINet code.")
    return warnings


# Convenience lookup maps -----------------------------------------------------
_CW = build_crosswalk()
FIPS_TO_ABBREV = dict(zip(_CW["fips"], _CW["abbrev_upper"]))
ABBREV_TO_FIPS = dict(zip(_CW["abbrev_upper"], _CW["fips"]))
FIPS_TO_NAME = dict(zip(_CW["fips"], _CW["state_name"]))
FIPS_TO_ILINET = dict(zip(_CW["fips"], _CW["abbrev_lower"]))


def fips_to_ilinet(fips: str) -> Optional[str]:
    """ILINet (fluview) region code for a FluSight FIPS, or None if unavailable."""
    return FIPS_TO_ILINET.get(str(fips))


def write_csv(path: str = "data/crosswalk.csv") -> str:
    """Write the crosswalk to ``path`` and return the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    build_crosswalk().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import pandas as pd  # noqa: F811

    cw = build_crosswalk()
    out = write_csv()
    print(f"Wrote {out} ({len(cw)} rows).")

    # Validate against the actual raw files when present.
    try:
        fl = pd.read_csv("data/raw/flusight_truth.csv")["location"].astype(str).unique()
        nh = pd.read_csv("data/raw/nhsn.csv")["jurisdiction"].astype(str).unique()
        il = pd.read_csv("data/raw/ilinet.csv")["region"].astype(str).unique()
        print("\nValidating crosswalk against raw data...")
        validate(cw, fl, nh, il)
    except FileNotFoundError as e:
        print(f"  (skipping validation; raw file missing: {e})")

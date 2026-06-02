"""Characterise the NHSN voluntary -> mandatory reporting shift (Nov 2024).

NHSN hospital respiratory reporting was *voluntary* until 2024-11-01, then became
*mandatory*. Reported flu-admission counts (``totalconfflunewadm``) can jump at
that boundary purely because more hospitals report, not because incidence
changed — a confounder any forecaster must be aware of.

This module:

* plots weekly ``totalconfflunewadm`` per jurisdiction with the mandate date
  marked, saving ``figures/nhsn_reporting_shift.png``;
* computes mean +/- std before and after the mandate per jurisdiction;
* prints jurisdictions whose mean changed by >20% post-mandate;
* writes the processed NHSN frame with a boolean ``post_mandatory`` column to
  ``data/processed/nhsn_clean.csv`` (so the flag is available as a covariate).

Run with ``python -m ingestion.nhsn_shift``.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.nhsn import NHSN_MANDATE_DATE  # noqa: E402

RAW_PATH = "data/raw/nhsn.csv"
PROCESSED_PATH = "data/processed/nhsn_clean.csv"
FIG_PATH = "figures/nhsn_reporting_shift.png"
# HHS rollup rows are not jurisdictions; drop them from the per-state analysis.
_NON_STATE = {f"Region {i}" for i in range(1, 11)}


def load_nhsn(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw NHSN flu admissions -> [reference_date, location, value, post_mandatory]."""
    df = pd.read_csv(path, usecols=["weekendingdate", "jurisdiction", "totalconfflunewadm"])
    df = df.rename(columns={"weekendingdate": "reference_date",
                            "jurisdiction": "location",
                            "totalconfflunewadm": "value"})
    df["reference_date"] = pd.to_datetime(df["reference_date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[~df["location"].isin(_NON_STATE)].copy()
    df["post_mandatory"] = df["reference_date"] >= NHSN_MANDATE_DATE
    return df.sort_values(["location", "reference_date"]).reset_index(drop=True)


def pre_post_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of weekly admissions before vs after the mandate, per jurisdiction."""
    rows = []
    for loc, g in df.groupby("location"):
        pre = g.loc[~g["post_mandatory"], "value"]
        post = g.loc[g["post_mandatory"], "value"]
        if len(pre) == 0 or len(post) == 0:
            continue
        pre_m, post_m = pre.mean(), post.mean()
        pct = 100.0 * (post_m - pre_m) / pre_m if pre_m else np.nan
        rows.append({"location": loc, "pre_mean": pre_m, "pre_std": pre.std(),
                     "post_mean": post_m, "post_std": post.std(), "pct_change": pct})
    return pd.DataFrame(rows).sort_values("pct_change", ascending=False).reset_index(drop=True)


def plot_shift(df: pd.DataFrame, out_path: str = FIG_PATH) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    for loc, g in df.groupby("location"):
        lw, color, alpha = (2.2, "black", 1.0) if loc == "USA" else (0.6, None, 0.35)
        ax.plot(g["reference_date"], g["value"], lw=lw, color=color, alpha=alpha,
                label="USA (national)" if loc == "USA" else None)
    ax.axvline(NHSN_MANDATE_DATE, color="tab:red", ls="--", lw=1.5,
               label="Mandatory reporting (2024-11-01)")
    ax.set(title="NHSN weekly confirmed-flu admissions by jurisdiction\n"
                 "(voluntary -> mandatory reporting shift)",
           xlabel="Week ending", ylabel="Confirmed flu admissions")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run():
    df = load_nhsn()
    fig = plot_shift(df)
    print(f"Saved {fig}")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote processed NHSN with post_mandatory column -> {PROCESSED_PATH} "
          f"({len(df)} rows, {df['post_mandatory'].sum()} post-mandate)")

    stats = pre_post_stats(df)
    big = stats[stats["pct_change"].abs() > 20.0]
    print(f"\nJurisdictions with >20% mean change post-mandate ({len(big)} of {len(stats)}):")
    print(big.round(1).to_string(index=False))
    return df, stats


if __name__ == "__main__":
    run()

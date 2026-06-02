"""Spatial analysis: does MIST help most in high-mobility hub states? (Phase 4.3)

Uses the **gravity mobility matrix** (`features.mobility`) as an independent
reference for connectivity (MIST's own spatial prior is the correlation proxy; see
the Phase 4 note in CHANGES.md). States are split into connectivity strata by total
gravity outflow, and we compare MIST's WIS improvement over ARIMA across strata and
territories. The result is drawn as a centroid bubble map (no shapefile needed).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.mobility import GEO, total_outflow


def connectivity_strata(locations) -> pd.DataFrame:
    """Per-location total gravity outflow + connectivity tier (high/low/territory)."""
    outflow = total_outflow(list(locations))
    rows = []
    for loc, flow in outflow.items():
        is_terr = loc in {"72", "60", "66", "69", "78"}
        rows.append({"location": loc, "outflow": flow, "territory": is_terr})
    df = pd.DataFrame(rows)
    states = df[(~df["territory"]) & (df["location"] != "US")]
    hi_cut = states["outflow"].quantile(0.75)
    lo_cut = states["outflow"].quantile(0.25)

    def tier(r):
        if r["territory"]:
            return "territory"
        if r["location"] == "US":
            return "national"
        if r["outflow"] >= hi_cut:
            return "high"
        if r["outflow"] <= lo_cut:
            return "low"
        return "mid"

    df["tier"] = df.apply(tier, axis=1)
    return df


def improvement_by_stratum(results: pd.DataFrame, focal: str = "mist_v2",
                           ref: str = "arima", metric: str = "wis") -> pd.DataFrame:
    """Mean per-location (ref - focal) improvement, joined to connectivity tier."""
    piv = (results[results["model"].isin([focal, ref])]
           .groupby(["model", "location"])[metric].mean().unstack("model"))
    piv = piv.dropna(subset=[focal, ref])
    piv["improvement"] = piv[ref] - piv[focal]          # positive => focal better
    strata = connectivity_strata(piv.index.tolist()).set_index("location")
    out = piv.join(strata)
    return out.reset_index()


def spatial_summary(imp: pd.DataFrame) -> pd.DataFrame:
    """Mean MIST-over-ARIMA improvement by connectivity tier."""
    return (imp.groupby("tier")["improvement"]
            .agg(["mean", "median", "count"]).reset_index()
            .sort_values("mean", ascending=False))


def connectivity_map(imp: pd.DataFrame, out_path: str, focal: str = "mist_v2",
                     ref: str = "arima") -> str:
    """Centroid bubble map: bubble size = |improvement|, colour = sign."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for _, r in imp.iterrows():
        loc = r["location"]
        if loc not in GEO or loc == "US":
            continue
        lat, lon, _ = GEO[loc]
        val = r["improvement"]
        ax.scatter(lon, lat, s=30 + min(abs(val), 600) * 0.6,
                   c=("tab:green" if val > 0 else "tab:red"), alpha=0.6,
                   edgecolors="k", linewidths=0.4)
        ax.text(lon, lat, loc, fontsize=6, ha="center", va="center")
    ax.set(title=f"{focal} WIS improvement over {ref} by location "
                 f"(green = MIST better; bubble size = |WIS gap|)",
           xlabel="Longitude", ylabel="Latitude", xlim=(-170, -65), ylim=(15, 65))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

"""Phase 5 — paper-ready figures.

* :func:`main_result_figure` (5.1) — 3-panel (Rising/Peak/Declining) WIS bar chart,
  MIST highlighted, ARIMA reference line, DM-significance asterisks, colorblind-safe
  palette. Saved as vector PDF + 300-dpi PNG.
* :func:`attention_map_figure` (5.2) — for a rising-phase week in a hub state, the
  learned **spatial** mechanistic-attention weights drawn as a centroid edge map and
  compared against the independent **gravity** mobility flows (the falsifiable
  "does attention align with mobility?" check).
* :func:`case_study_figure` (5.3) — rising-phase episodes where MIST gave early
  warning and ARIMA did not: actuals + MIST vs ARIMA 50/95% bands.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colorblind-safe palette (Wong 2011).
CB = {"mist": "#0072B2", "arima": "#D55E00", "other": "#999999",
      "rising": "#E69F00", "band": "#56B4E9"}
PHASES = ["Rising", "Peak", "Declining"]


# ----------------------------------------------------------------- 5.1 main result
def main_result_figure(phase_perf: pd.DataFrame, out_stem: str,
                       dm_p: Optional[pd.DataFrame] = None,
                       focal: str = "mist_v2", ref: str = "arima") -> str:
    """3-panel WIS bars by phase. ``phase_perf`` indexed by model, columns = phases."""
    models = [m for m in phase_perf.index if not m.startswith("mist_no")]
    order = sorted(models, key=lambda m: (m != focal, m))   # focal first
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, phase in zip(axes, PHASES):
        vals = phase_perf.loc[order, phase]
        colors = [CB["mist"] if m == focal else (CB["arima"] if m == ref else CB["other"])
                  for m in order]
        bars = ax.bar(range(len(order)), vals.values, color=colors)
        if ref in phase_perf.index:
            ax.axhline(phase_perf.loc[ref, phase], color=CB["arima"], ls="--", lw=1,
                       label=f"{ref} ref")
        # DM-significance asterisks: focal vs each model in this phase.
        if dm_p is not None:
            for i, m in enumerate(order):
                if m == focal or focal not in dm_p.index or m not in dm_p.columns:
                    continue
                p = dm_p.loc[focal, m]
                star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                if star:
                    ax.text(i, vals.values[i], star, ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{phase} phase")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean WIS")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Phase-stratified WIS (lower is better; ** p<0.01, * p<0.05 vs MIST)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=300)
    plt.close(fig)
    return out_stem + ".pdf"


# ------------------------------------------------------------- 5.2 attention map
def attention_map_figure(model, store, *, signal: str, source: str, focal_loc: str,
                         origin, out_path: str, top_k: int = 10) -> Optional[str]:
    """Draw learned spatial attention from ``focal_loc`` vs gravity mobility flows."""
    from features.mobility import GEO, gravity_matrix

    net = getattr(model, "net", None)
    if net is None or focal_loc not in model.panel_locations:
        return None
    # One forward pass at the origin populates spatial attention weights.
    model._cache.clear()
    model._forward_panel(pd.Timestamp(origin))
    attn_mod = getattr(net.spatial_block, "attn", None)
    if attn_mod is None or attn_mod.last_attn is None:
        return None
    # last_attn: (B*T, h, S, S) -> average over batch*tokens and heads.
    A = attn_mod.last_attn.mean(dim=(0, 1)).cpu().numpy()    # (S,S)
    locs = list(model.panel_locations)
    fi = locs.index(focal_loc)
    weights = A[fi]                                          # attention focal -> others
    W = gravity_matrix(locs)
    grav = W[fi]

    order = np.argsort(weights)[::-1]
    order = [j for j in order if locs[j] != focal_loc and locs[j] in GEO][:top_k]

    fig, ax = plt.subplots(figsize=(12, 7))
    flat, flon, _ = GEO.get(focal_loc, (39.5, -98.5, 0))
    # Gravity flows (grey) for reference.
    g_top = [j for j in np.argsort(grav)[::-1] if locs[j] != focal_loc and locs[j] in GEO][:top_k]
    for j in g_top:
        lat, lon, _ = GEO[locs[j]]
        ax.plot([flon, lon], [flat, lat], color="grey", lw=0.8 + 3 * grav[j], alpha=0.35,
                zorder=1)
    # Learned attention edges (blue).
    wmax = max(weights[order].max(), 1e-9)
    for j in order:
        lat, lon, _ = GEO[locs[j]]
        ax.plot([flon, lon], [flat, lat], color=CB["mist"],
                lw=0.6 + 4 * weights[j] / wmax, alpha=0.8, zorder=2)
        ax.text(lon, lat, locs[j], fontsize=7, ha="center", va="center")
    ax.scatter([flon], [flat], s=120, color=CB["rising"], edgecolors="k", zorder=3)
    ax.text(flon, flat, focal_loc, fontsize=9, ha="center", va="center", zorder=4)
    ax.set(title=f"Learned spatial attention from {focal_loc} (blue) vs gravity mobility "
                 f"(grey), rising-phase week {pd.Timestamp(origin).date()}",
           xlabel="Longitude", ylabel="Latitude", xlim=(-170, -65), ylim=(15, 65))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- 5.3 case study
def case_study_figure(results_all: pd.DataFrame, store, mist_model, arima_model, *,
                      signal: str, source: str, out_path: str,
                      horizon: int = 2, n_episodes: int = 2) -> Optional[str]:
    """Rising episodes where MIST beat ARIMA most: actuals + MIST/ARIMA bands."""
    # Find (location, forecast_date) where MIST's WIS advantage over ARIMA is largest
    # *during the rising phase* (genuine early-warning episodes).
    from evaluation.visualizer import attach_phase
    panel = results_all["location"].astype(str).unique().tolist()
    truth = pd.concat([store.get_vintage(signal, l, "2100-01-01", source=source)
                       for l in panel], ignore_index=True)
    labelled = attach_phase(results_all, truth)
    rising = labelled[(labelled["phase"] == "Rising") &
                      (labelled["horizon"] == horizon) &
                      (labelled["model"].isin(["mist_v2", "arima"]))]
    piv = rising.pivot_table(index=["location", "forecast_date"], columns="model", values="wis")
    piv = piv.dropna()
    if piv.empty:
        return None
    piv["gain"] = piv["arima"] - piv["mist_v2"]
    top = piv.sort_values("gain", ascending=False).head(n_episodes).index.tolist()

    fig, axes = plt.subplots(1, len(top), figsize=(7 * len(top), 5), squeeze=False)
    for ax, (loc, fdate) in zip(axes[0], top):
        truth = store.get_vintage(signal, loc, "2100-01-01", source=source) \
            .sort_values("reference_date")
        win = truth[(truth["reference_date"] >= pd.Timestamp(fdate) - pd.Timedelta(weeks=8)) &
                    (truth["reference_date"] <= pd.Timestamp(fdate) + pd.Timedelta(weeks=6))]
        ax.plot(win["reference_date"], win["value"], color="black", lw=1.8,
                label="Observed", zorder=5)
        hist = store.get_vintage(signal, loc, fdate, source=source)
        for mdl, color, name in [(mist_model, CB["mist"], "MIST"),
                                 (arima_model, CB["arima"], "ARIMA")]:
            p = mdl.predict(hist, fdate, horizons=(1, 2, 3, 4))
            if p.empty:
                continue
            pv = p.pivot_table(index="reference_date", columns="quantile", values="value")
            x = pv.index
            ax.fill_between(x, pv[0.025], pv[0.975], color=color, alpha=0.12)
            ax.fill_between(x, pv[0.25], pv[0.75], color=color, alpha=0.25)
            ax.plot(x, pv[0.5], color=color, lw=1.5, marker="o", ms=3, label=name)
        ax.axvline(pd.Timestamp(fdate), color="grey", ls=":", lw=0.8)
        ax.set_title(f"{loc}, origin {pd.Timestamp(fdate).date()}")
        ax.set_xlabel("Week"); ax.set_ylabel("Flu hospital admissions")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    fig.suptitle("Rising-phase early warning: MIST vs ARIMA 50/95% intervals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------- multi-season comparison
def season_comparison_figure(season_wis: pd.DataFrame, out_stem: str,
                             focal: str = "mist_v2") -> str:
    """Grouped WIS bars by season x model from ``results/season_wis.csv``.

    ``season_wis`` columns: season, model, wis (others ignored). MIST is highlighted
    in the Wong-palette blue, ARIMA in orange, others grey. The thin-data 2022-23
    season is annotated on the MIST bar ("~38 training weeks") to explain its weaker
    result honestly rather than dropping it.
    """
    piv = season_wis.pivot(index="season", columns="model", values="wis")
    seasons = list(piv.index)
    models = [focal] + [m for m in piv.columns if m != focal]
    x = np.arange(len(seasons))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(models):
        color = CB["mist"] if m == focal else (CB["arima"] if m == "arima" else CB["other"])
        offs = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offs, piv[m].values, width, label=m, color=color,
               edgecolor="black", linewidth=0.4)

    # Honest annotation on the thin-data 2022-23 MIST bar.
    if "2022-23" in seasons and focal in piv.columns:
        si = seasons.index("2022-23")
        foffs = (models.index(focal) - (len(models) - 1) / 2) * width
        ax.annotate("~38 training weeks", xy=(x[si] + foffs, piv.loc["2022-23", focal]),
                    xytext=(x[si] + foffs, piv.loc["2022-23", focal] + 18),
                    ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"))

    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set(xlabel="Flu season", ylabel="Mean WIS",
           title="Cross-season WIS by model (lower is better)")
    ax.legend(fontsize=8, ncol=len(models)); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=300)
    plt.close(fig)
    return out_stem + ".pdf"

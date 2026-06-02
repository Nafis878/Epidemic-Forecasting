"""Phase 4.1-4.3 — significance, horizon breakdown, and spatial analysis.

Consumes ``results/results_all_v2.csv`` from ``run_benchmark_v2`` and produces:

* ``tables/dm_significance.csv``       — Diebold-Mariano (HLN) p-value matrix (4.1),
* ``tables/dm_headline_rising.csv``    — MIST vs all in the rising phase (effect
  size + CI + p),
* ``results/horizon_breakdown.csv`` + ``figures/horizon_breakdown.png`` (4.2),
* ``figures/spatial_connectivity.png`` + ``results/spatial_strata.csv`` (4.3).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dm_test import dm_matrix, headline_report  # noqa: E402
from evaluation.spatial import (connectivity_map, improvement_by_stratum,  # noqa: E402
                                spatial_summary)
from evaluation.visualizer import attach_phase  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES, TAB, FIG = (os.path.join(ROOT, d) for d in ("results", "tables", "figures"))
SIG, SRC = "flu_hosp_admissions", "flusight"


def _load():
    df = pd.read_csv(os.path.join(RES, "results_all_v2.csv"),
                     parse_dates=["forecast_date", "reference_date"])
    store = VersionedStore()
    panel = df["location"].astype(str).unique().tolist()
    truth = pd.concat([store.get_vintage(SIG, l, "2100-01-01", source=SRC) for l in panel],
                      ignore_index=True)
    return attach_phase(df, truth)


def run():
    for d in (TAB, FIG, RES):
        os.makedirs(d, exist_ok=True)
    df = _load()
    models = sorted(df["model"].unique())

    # --- 4.1 DM significance ----------------------------------------------
    P = dm_matrix(df, models=models, metric="wis")
    P.to_csv(os.path.join(TAB, "dm_significance.csv"))
    print("=== Diebold-Mariano p-values (WIS, HLN-corrected) ===")
    print(P.round(3).to_string())

    head = headline_report(df, focal="mist_v2", metric="wis", phase="Rising")
    head.to_csv(os.path.join(TAB, "dm_headline_rising.csv"), index=False)
    print("\n=== mist_v2 vs all, RISING phase (mean WIS diff; <0 => MIST better) ===")
    print(head.round(3).to_string(index=False))

    # --- 4.2 horizon breakdown --------------------------------------------
    hb = df.groupby(["model", "horizon"])["wis"].mean().unstack("horizon")
    hb.to_csv(os.path.join(RES, "horizon_breakdown.csv"))
    print("\n=== Mean WIS by model x horizon ===")
    print(hb.round(1).to_string())
    _horizon_fig(hb, os.path.join(FIG, "horizon_breakdown.png"))

    # --- 4.3 spatial connectivity -----------------------------------------
    if "arima" in models and "mist_v2" in models:
        imp = improvement_by_stratum(df, focal="mist_v2", ref="arima")
        strata = spatial_summary(imp)
        strata.to_csv(os.path.join(RES, "spatial_strata.csv"), index=False)
        connectivity_map(imp, os.path.join(FIG, "spatial_connectivity.png"))
        print("\n=== MIST-over-ARIMA WIS improvement by connectivity tier ===")
        print(strata.round(1).to_string(index=False))
    print(f"\nFigures -> {FIG}; tables -> {TAB}")


def _horizon_fig(hb: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in hb.index:
        ax.plot(hb.columns, hb.loc[model], marker="o", label=model)
    ax.set(title="WIS by forecast horizon (lower is better)",
           xlabel="Horizon (weeks ahead)", ylabel="Mean WIS")
    ax.set_xticks(list(hb.columns))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()

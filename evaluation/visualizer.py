"""Visualisation & phase analysis for the MIST-Transformer benchmark.

Provides:

* :func:`label_phases` — classify each observed week into **Rising / Peak /
  Declining** from the (final) truth trajectory, so forecasts can be analysed by
  epidemic phase.
* :func:`fan_chart` — a forecast "fan chart": observed truth plus the model's
  predictive median and 50% / 95% quantile bands at successive origins.
* :func:`phase_performance_plot` — grouped bars of mean WIS by model x phase,
  the figure used to answer whether mechanistic attention helps in the rising
  phase.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASES = ["Rising", "Peak", "Declining"]
_PHASE_COLORS = {"Rising": "tab:orange", "Peak": "tab:red", "Declining": "tab:green"}


# --------------------------------------------------------------------- phases
def label_phases(truth: pd.DataFrame, peak_frac: float = 0.85,
                 smooth: int = 3, season_window: int = 21) -> pd.DataFrame:
    """Label each (location, reference_date) as Rising / Peak / Declining.

    Heuristic on the smoothed truth curve per location, using a **local
    seasonal** maximum (rolling window) so the label reflects the current wave
    rather than an all-time peak from a different season:
      * **Peak**     — value within ``peak_frac`` of the local seasonal max.
      * **Rising**   — below peak and increasing.
      * **Declining**— below peak and non-increasing.

    Parameters
    ----------
    truth:
        Long DataFrame with columns ``location``, ``reference_date``, ``value``.
    season_window:
        Width (in weeks) of the centered rolling window defining the local
        seasonal maximum.

    Returns
    -------
    DataFrame with an added ``phase`` column.
    """
    out = []
    for loc, g in truth.sort_values("reference_date").groupby("location"):
        g = g.copy()
        v = g["value"].rolling(smooth, center=True, min_periods=1).mean()
        local_max = v.rolling(season_window, center=True, min_periods=5).max()
        slope = v.diff().fillna(0.0)
        phase = np.where(
            v >= peak_frac * local_max, "Peak",
            np.where(slope > 0, "Rising", "Declining"),
        )
        g["phase"] = phase
        out.append(g)
    return pd.concat(out, ignore_index=True)


def attach_phase(results: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join a phase label onto each result row by (location, reference_date)."""
    labelled = label_phases(truth)[["location", "reference_date", "phase"]]
    labelled["location"] = labelled["location"].astype(str)
    labelled["reference_date"] = pd.to_datetime(labelled["reference_date"])
    res = results.copy()
    res["location"] = res["location"].astype(str)
    res["reference_date"] = pd.to_datetime(res["reference_date"])
    return res.merge(labelled, on=["location", "reference_date"], how="left")


# ------------------------------------------------------------------ fan chart
def fan_chart(model, store, *, signal: str, source: str, location: str,
              origins: Sequence, horizons: Sequence[int] = (1, 2, 3, 4),
              title: Optional[str] = None, out_path: Optional[str] = None,
              final_as_of: str = "2100-01-01"):
    """Plot observed truth with the model's predictive fan at each origin."""
    truth = store.get_vintage(signal, location, final_as_of, source=source) \
        .sort_values("reference_date")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(truth["reference_date"], truth["value"], color="black", lw=1.6,
            label="Observed", zorder=5)

    first = True
    for origin in origins:
        origin = pd.Timestamp(origin)
        hist = store.get_vintage(signal, location, origin, source=source)
        if hist.empty:
            continue
        preds = model.predict(hist, origin, horizons=horizons)
        if preds.empty:
            continue
        piv = preds.pivot_table(index="reference_date", columns="quantile",
                                values="value").sort_index()
        # Anchor the fan at the last observed value for visual continuity.
        last_obs_date = hist.sort_values("reference_date")["reference_date"].iloc[-1]
        last_obs_val = hist.sort_values("reference_date")["value"].iloc[-1]
        x = [last_obs_date] + list(piv.index)

        def col(q):
            return [last_obs_val] + list(piv[q].values)

        ax.fill_between(x, col(0.025), col(0.975), color="tab:blue", alpha=0.12,
                        label="95% PI" if first else None, zorder=2)
        ax.fill_between(x, col(0.25), col(0.75), color="tab:blue", alpha=0.28,
                        label="50% PI" if first else None, zorder=3)
        ax.plot(x, col(0.5), color="tab:blue", lw=1.2,
                label="Median forecast" if first else None, zorder=4)
        ax.axvline(origin, color="grey", ls=":", lw=0.6, zorder=1)
        first = False

    ax.set(title=title or f"Fan chart — {location}", xlabel="Date",
           ylabel="Weekly flu hospital admissions")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    return out_path


# --------------------------------------------------------- phase performance
def phase_performance_plot(results_with_phase: pd.DataFrame, out_path: str,
                           models: Optional[Sequence[str]] = None,
                           metric: str = "wis"):
    """Grouped bar chart of mean ``metric`` by model within each phase."""
    df = results_with_phase.dropna(subset=["phase"])
    if models is not None:
        df = df[df["model"].isin(models)]
    table = df.groupby(["phase", "model"])[metric].mean().unstack("model")
    table = table.reindex(PHASES)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    model_order = list(table.columns)
    x = np.arange(len(PHASES))
    width = 0.8 / max(1, len(model_order))
    for i, m in enumerate(model_order):
        ax.bar(x + i * width, table[m].values, width, label=m)
    ax.set_xticks(x + width * (len(model_order) - 1) / 2)
    ax.set_xticklabels(PHASES)
    ax.set(title=f"Phase-wise performance (mean {metric.upper()}, lower is better)",
           xlabel="Epidemic phase", ylabel=f"Mean {metric.upper()}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return table

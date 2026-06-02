"""Calibration diagnostics: reliability diagrams and Expected Calibration Error.

A probabilistic forecast is *calibrated* if, for every quantile level ``tau``,
the observed value falls at or below the predicted ``tau``-quantile a fraction
``tau`` of the time. We measure this directly from model forecasts:

* :func:`collect_quantile_records` re-runs a model over rolling origins (the same
  strict-vintage path the back-tester uses) and keeps the **full** quantile vector
  plus the realised value for each (origin, horizon).
* :func:`reliability` computes empirical coverage per nominal level.
* :func:`ece` is the mean absolute gap between empirical and nominal coverage.
* :func:`reliability_plot` saves the diagram for several models.

These power Task 2.1 (diagnose) and verify Task 2.2 (the conformal fix).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.backtester import _FINAL_AS_OF, _final_truth_map
from evaluation.metrics import DEFAULT_QUANTILES

NOMINAL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def collect_quantile_records(store, model, forecast_dates, *, signal: str,
                             source: str, locations: Sequence[str],
                             horizons: Sequence[int] = (1, 2, 3, 4),
                             quantiles: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Long frame of every predicted quantile vs the realised value.

    Columns: ``location, forecast_date, reference_date, horizon, quantile,
    pred, y_true``. Uses ``get_vintage`` for inputs (no leakage).
    """
    quantiles = list(quantiles) if quantiles is not None else list(DEFAULT_QUANTILES)
    records = []
    for loc in locations:
        truth_map = _final_truth_map(store, signal, loc, source)
        for fdate in forecast_dates:
            fdate = pd.Timestamp(fdate)
            hist = store.get_vintage(signal, loc, fdate, source=source)
            if hist.empty:
                continue
            preds = model.predict(history=hist, forecast_date=fdate,
                                  horizons=horizons, quantiles=quantiles)
            if preds.empty:
                continue
            for _, r in preds.iterrows():
                y = truth_map.get(pd.Timestamp(r["reference_date"]))
                if y is None or (isinstance(y, float) and np.isnan(y)):
                    continue
                records.append({"location": loc, "forecast_date": fdate,
                                "reference_date": pd.Timestamp(r["reference_date"]),
                                "horizon": int(r["horizon"]),
                                "quantile": float(r["quantile"]),
                                "pred": float(r["value"]), "y_true": float(y)})
    return pd.DataFrame.from_records(records)


def reliability(records: pd.DataFrame,
                levels: Sequence[float] = NOMINAL_LEVELS) -> pd.DataFrame:
    """Empirical coverage ``P(y <= q_tau)`` per nominal level ``tau``."""
    rows = []
    for tau in levels:
        sub = records[np.isclose(records["quantile"], tau)]
        if sub.empty:
            continue
        emp = float((sub["y_true"] <= sub["pred"]).mean())
        rows.append({"nominal": tau, "empirical": emp, "n": len(sub)})
    return pd.DataFrame(rows)


def ece(records: pd.DataFrame, levels: Sequence[float] = NOMINAL_LEVELS) -> float:
    """Expected Calibration Error: mean |empirical - nominal| over quantile levels."""
    rel = reliability(records, levels)
    if rel.empty:
        return float("nan")
    return float((rel["empirical"] - rel["nominal"]).abs().mean())


def ece_by_phase(records: pd.DataFrame, truth: pd.DataFrame,
                 levels: Sequence[float] = NOMINAL_LEVELS) -> pd.DataFrame:
    """ECE split by epidemic phase (Rising/Peak/Declining)."""
    from evaluation.visualizer import attach_phase
    labelled = attach_phase(records.rename(columns={}), truth)
    rows = []
    for phase, g in labelled.dropna(subset=["phase"]).groupby("phase"):
        rows.append({"phase": phase, "ece": ece(g, levels), "n": len(g)})
    return pd.DataFrame(rows)


def reliability_plot(records_by_model: dict, out_path: str,
                     levels: Sequence[float] = NOMINAL_LEVELS) -> str:
    """Reliability diagram overlaying several models against the diagonal."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, recs in records_by_model.items():
        rel = reliability(recs, levels)
        if rel.empty:
            continue
        ax.plot(rel["nominal"], rel["empirical"], marker="o",
                label=f"{name} (ECE={ece(recs, levels):.3f})")
    ax.set(xlabel="Nominal quantile level", ylabel="Empirical coverage",
           title="Reliability diagram (closer to diagonal = better calibrated)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

"""STEP 6 — comprehensive benchmark, summary tables, and analysis figures.

Consolidates every model's per-forecast results, adds a **MIST ablation**
(mechanistic attention switched OFF) to isolate the contribution of the
mechanistic prior, then:

* writes ``evaluation/results_all.csv``       (every scored forecast, all models),
* writes ``evaluation/results_summary.csv``   (WIS / MAE / coverage per model),
* writes ``evaluation/results_phase.csv``     (WIS per model x epidemic phase),
* saves fan charts + the phase-wise performance plot under ``evaluation/figures/``,
* prints whether mechanistic attention improved the *rising* phase.

Re-run STEP 3/4/5 runners first if their CSVs are missing.

Note on baselines: the brief lists "Prophet"; Prophet is not installed (heavy to
build on Windows) so the statistical baseline is the ARIMA model from STEP 3
(``arima_stat``) — an equivalent strong statistical baseline.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import rolling_origin_backtest, summarize  # noqa: E402
from evaluation.visualizer import (attach_phase, fan_chart,  # noqa: E402
                                    phase_performance_plot)
from features.versioned_store import VersionedStore  # noqa: E402
from models.mist_transformer import MISTModel  # noqa: E402

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "figures")
SIGNAL, SOURCE = "flu_hosp_admissions", "flusight"
EVAL_LOCATIONS = ["US", "06", "36"]
HORIZONS = (1, 2, 3, 4)
TRAIN_END = "2023-10-28"
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")

EXISTING = {
    "results_baselines.csv": ["arima_stat", "seir_mechanistic"],
    "results_dl.csv": ["tft", "patchtst"],
    "results_mist.csv": ["mist"],
}


def _all_flusight_locations(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SOURCE) & (sig["signal"] == SIGNAL)]["location"].tolist()


def _load_existing():
    frames = []
    for fname in EXISTING:
        p = os.path.join(HERE, fname)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, parse_dates=["forecast_date", "reference_date"]))
        else:
            print(f"  (missing {fname} — re-run its STEP runner to include it)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_ablation(store):
    """Train MIST with mechanistic attention OFF and back-test it."""
    panel = _all_flusight_locations(store)
    model = MISTModel(epochs=80, seed=0, use_mechanistic=False)
    t0 = time.time()
    model.fit(store, signal=SIGNAL, source=SOURCE, locations=panel,
              train_end_date=TRAIN_END, verbose=False)
    print(f"  ablation (mist_no_mech) trained in {time.time() - t0:.1f}s")
    frames = []
    for loc in EVAL_LOCATIONS:
        res = rolling_origin_backtest(store, model, ORIGINS, signal=SIGNAL,
                                      location=loc, source=SOURCE, horizons=HORIZONS)
        res.insert(0, "model", "mist_no_mech")
        frames.append(res)
    return pd.concat(frames, ignore_index=True), model


def run():
    os.makedirs(FIG_DIR, exist_ok=True)
    store = VersionedStore()

    print("Loading existing model results...")
    existing = _load_existing()
    print("Training MIST ablation (mechanistic attention OFF)...")
    ablation, _ = _run_ablation(store)

    results = pd.concat([existing, ablation], ignore_index=True)
    results.to_csv(os.path.join(HERE, "results_all.csv"), index=False)

    # --- per-model summary (WIS / MAE / coverage) --------------------------
    summary = (results.groupby("model")
               .agg(n=("wis", "size"), wis=("wis", "mean"), mae=("mae", "mean"),
                    coverage_50=("cov_50", "mean"), coverage_95=("cov_95", "mean"))
               .sort_values("wis").reset_index())
    summary.to_csv(os.path.join(HERE, "results_summary.csv"), index=False)
    print("\n=== results_summary.csv (overall, lower WIS is better) ===")
    print(summary.round(3).to_string(index=False))

    # --- phase analysis ----------------------------------------------------
    truth = pd.concat([
        store.get_vintage(SIGNAL, loc, "2100-01-01", source=SOURCE)
        for loc in EVAL_LOCATIONS
    ], ignore_index=True)
    res_phase = attach_phase(results, truth)

    phase_tbl = (res_phase.dropna(subset=["phase"])
                 .groupby(["model", "phase"])["wis"].mean()
                 .unstack("phase").reindex(columns=["Rising", "Peak", "Declining"]))
    phase_tbl.to_csv(os.path.join(HERE, "results_phase.csv"))
    print("\n=== results_phase.csv (mean WIS by phase) ===")
    print(phase_tbl.round(1).to_string())

    # --- figures -----------------------------------------------------------
    phase_performance_plot(res_phase, os.path.join(FIG_DIR, "phase_performance.png"))
    # Fan chart needs full quantiles -> regenerate from the (re-trained) MIST.
    mist = MISTModel(epochs=80, seed=0, use_mechanistic=True)
    mist.fit(store, signal=SIGNAL, source=SOURCE,
             locations=_all_flusight_locations(store), train_end_date=TRAIN_END)
    fan_origins = pd.date_range("2023-11-11", "2024-03-30", freq="28D")
    fan_chart(mist, store, signal=SIGNAL, source=SOURCE, location="US",
              origins=fan_origins, title="MIST-Transformer fan chart — US national",
              out_path=os.path.join(FIG_DIR, "fan_chart_US.png"))
    fan_chart(mist, store, signal=SIGNAL, source=SOURCE, location="06",
              origins=fan_origins, title="MIST-Transformer fan chart — California",
              out_path=os.path.join(FIG_DIR, "fan_chart_CA.png"))
    print(f"\nFigures saved to {FIG_DIR}")

    # --- answer the research question --------------------------------------
    _analyse_rising_phase(phase_tbl)
    return results, summary, phase_tbl


def _analyse_rising_phase(phase_tbl: pd.DataFrame):
    print("\n" + "=" * 64)
    print("ANALYSIS: did Mechanistic Attention help the RISING phase?")
    print("=" * 64)
    if not {"mist", "mist_no_mech"}.issubset(phase_tbl.index):
        print("  (need both 'mist' and 'mist_no_mech' rows)")
        return
    for phase in ["Rising", "Peak", "Declining"]:
        full = phase_tbl.loc["mist", phase]
        abl = phase_tbl.loc["mist_no_mech", phase]
        delta = abl - full
        pct = 100 * delta / abl if abl else float("nan")
        verdict = "IMPROVED" if delta > 0 else "no improvement"
        print(f"  {phase:9s}: WIS full={full:7.1f}  no-mech={abl:7.1f}  "
              f"delta={delta:+7.1f} ({pct:+.1f}%)  -> mechanistic {verdict}")


if __name__ == "__main__":
    run()

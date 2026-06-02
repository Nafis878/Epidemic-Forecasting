"""Phase 4.0 — comprehensive full-panel benchmark (v2).

Trains every model, back-tests each over the **full ~53-location FluSight panel**
for the 2023-24 season, and writes:

* ``results/results_all_v2.csv``        — every scored forecast, all models
  (consumed by the DM test, horizon breakdown, and spatial analysis),
* ``results/leaderboard_v2.csv``        — WIS / MAE / cov-50 / cov-95 per model,
* ``results/phase_performance_v2.csv``  — mean WIS per model x epidemic phase.

Models include the full MIST v2 stack (correlation spatial prior + R_t blend +
ILINet analogue prior + ACI calibration) and its ablations, plus the ARIMA, SEIR,
TFT and PatchTST baselines.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import rolling_origin_backtest  # noqa: E402
from evaluation.visualizer import attach_phase  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402
from models.analogue_blend import AnalogueBlendModel  # noqa: E402
from models.baseline_mechanistic import SEIRModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402
from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
SIG, SRC = "flu_hosp_admissions", "flusight"
HORIZONS = (1, 2, 3, 4)
TRAIN_END = "2023-10-28"
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")


def _panel(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SRC) & (sig["signal"] == SIG)]["location"].tolist()


def _backtest_all(store, model, name, locations):
    frames = []
    for loc in locations:
        res = rolling_origin_backtest(store, model, ORIGINS, signal=SIG, location=loc,
                                      source=SRC, horizons=HORIZONS)
        if not res.empty:
            res.insert(0, "model", name)
            frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_mist(store, panel, **flags):
    m = MISTModelV2(epochs=80, seed=0, use_conformal=False, **flags)
    m.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)
    return m


def run():
    os.makedirs(RES, exist_ok=True)
    store = VersionedStore()
    panel = _panel(store)
    print(f"Full panel: {len(panel)} locations | {len(ORIGINS)} origins")

    def aci(m):
        return OnlineConformalModel(m, store, signal=SIG, source=SRC)

    print("Training MIST variants...")
    full = _fit_mist(store, panel, use_blend=True, use_mechanistic=True)
    no_blend = _fit_mist(store, panel, use_blend=False, use_mechanistic=True)
    no_mech = _fit_mist(store, panel, use_blend=True, use_mechanistic=False)
    print("Training DL baselines...")
    tft = TFTModel(epochs=40, seed=0)
    tft.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)
    patchtst = PatchTSTModel(epochs=40, seed=0)
    patchtst.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)

    models = {
        "mist_v2":          aci(AnalogueBlendModel(full)),          # full stack
        "mist_no_analogue": aci(full),
        "mist_no_blend":    aci(no_blend),
        "mist_no_mech":     aci(AnalogueBlendModel(no_mech)),
        "mist_no_aci":      AnalogueBlendModel(full),               # uncalibrated
        "arima":            ARIMAQuantileModel(),
        "seir":             SEIRModel(),
        "tft":              tft,
        "patchtst":         patchtst,
    }

    frames = []
    for name, model in models.items():
        t0 = time.time()
        res = _backtest_all(store, model, name, panel)
        frames.append(res)
        print(f"  {name:16s}: {len(res):5d} forecasts, WIS={res['wis'].mean():7.1f} "
              f"({time.time()-t0:.0f}s)")

    allres = pd.concat(frames, ignore_index=True)
    allres.to_csv(os.path.join(RES, "results_all_v2.csv"), index=False)

    leaderboard = (allres.groupby("model")
                   .agg(n=("wis", "size"), wis=("wis", "mean"), mae=("mae", "mean"),
                        cov_50=("cov_50", "mean"), cov_95=("cov_95", "mean"))
                   .sort_values("wis").reset_index())
    leaderboard.to_csv(os.path.join(RES, "leaderboard_v2.csv"), index=False)
    print("\n=== leaderboard_v2.csv (overall, lower WIS is better) ===")
    print(leaderboard.round(3).to_string(index=False))

    truth = pd.concat([store.get_vintage(SIG, loc, "2100-01-01", source=SRC) for loc in panel],
                      ignore_index=True)
    phase = (attach_phase(allres, truth).dropna(subset=["phase"])
             .groupby(["model", "phase"])["wis"].mean()
             .unstack("phase").reindex(columns=["Rising", "Peak", "Declining"]))
    phase.to_csv(os.path.join(RES, "phase_performance_v2.csv"))
    print("\n=== phase_performance_v2.csv (mean WIS by phase) ===")
    print(phase.round(1).to_string())
    return allres, leaderboard, phase


if __name__ == "__main__":
    run()

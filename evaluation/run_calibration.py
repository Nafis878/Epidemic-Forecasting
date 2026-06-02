"""Phase 2 — calibration diagnosis and the conformal fix.

Trains the baselines + MIST, builds the **ACI-calibrated** MIST, then:

* collects full-quantile forecast records on the evaluation locations,
* writes the reliability diagram ``figures/calibration_reliability.png`` with ECE
  per model (Task 2.1),
* prints per-model 50%/95% coverage and per-phase ECE for MIST, showing the
  conformal fix brings coverage to nominal (Task 2.2),
* saves ``results/calibrated_records.csv`` for downstream figures.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.calibration import (collect_quantile_records, ece,  # noqa: E402
                                    ece_by_phase, reliability_plot)
from features.versioned_store import VersionedStore  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402
from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
SIG, SRC = "flu_hosp_admissions", "flusight"
EVAL = ["US", "06", "36"]
HORIZONS = (1, 2, 3, 4)
TRAIN_END = "2023-10-28"
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")


def _panel(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SRC) & (sig["signal"] == SIG)]["location"].tolist()


def run():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    store = VersionedStore()
    panel = _panel(store)

    print("Training models for calibration diagnosis...")
    mist = MISTModelV2(epochs=80, seed=0, use_conformal=False)
    mist.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)
    tft = TFTModel(epochs=40, seed=0)
    tft.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)
    patchtst = PatchTSTModel(epochs=40, seed=0)
    patchtst.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)
    mist_aci = OnlineConformalModel(mist, store, signal=SIG, source=SRC)

    models = {"arima": ARIMAQuantileModel(), "tft": tft, "patchtst": patchtst,
              "mist": mist, "mist+ACI": mist_aci}

    records = {}
    for name, model in models.items():
        t0 = time.time()
        recs = collect_quantile_records(store, model, ORIGINS, signal=SIG, source=SRC,
                                        locations=EVAL, horizons=HORIZONS)
        records[name] = recs
        print(f"  {name:10s}: {len(recs)} quantile records ({time.time()-t0:.1f}s)")

    # --- reliability figure (Task 2.1) -------------------------------------
    out = reliability_plot(records, os.path.join(FIG, "calibration_reliability.png"))
    print(f"\nSaved {out}")

    # --- coverage + ECE table ----------------------------------------------
    print("\n=== Calibration summary (nominal 50% / 95%) ===")
    print(f"{'model':10s} {'cov50':>7s} {'cov95':>7s} {'ECE':>7s}")
    for name, recs in records.items():
        print(f"{name:10s} "
              f"{_central(recs, 0.25, 0.75):7.3f} "
              f"{_central(recs, 0.025, 0.975):7.3f} "
              f"{ece(recs):7.3f}")

    # --- per-phase ECE for MIST vs MIST+ACI (Task 2.2) ---------------------
    truth = pd.concat([store.get_vintage(SIG, loc, "2100-01-01", source=SRC) for loc in EVAL],
                      ignore_index=True)
    print("\n=== Per-phase ECE: MIST vs MIST+ACI ===")
    for name in ("mist", "mist+ACI"):
        ph = ece_by_phase(records[name], truth)
        print(f"\n{name}:")
        print(ph.round(3).to_string(index=False))

    # --- persist the calibrated records ------------------------------------
    for name in ("mist", "mist+ACI"):
        records[name].assign(model=name).to_csv(
            os.path.join(RES, f"calibrated_records_{name.replace('+','_')}.csv"), index=False)
    print(f"\nSaved calibrated records under {RES}/")
    return records


def _central(recs: pd.DataFrame, lo: float, hi: float) -> float:
    """Empirical coverage of the central interval [lo, hi] from quantile records."""
    piv = recs.pivot_table(index=["location", "reference_date", "horizon"],
                           columns="quantile", values="pred")
    yt = recs.groupby(["location", "reference_date", "horizon"])["y_true"].first()
    cols = piv.columns.to_numpy()
    import numpy as np
    lo_col = cols[np.argmin(np.abs(cols - lo))]
    hi_col = cols[np.argmin(np.abs(cols - hi))]
    inside = (yt.values >= piv[lo_col].values) & (yt.values <= piv[hi_col].values)
    return float(inside.mean())


if __name__ == "__main__":
    run()

"""Train the TFT and PatchTST baselines and evaluate them with the back-tester.

Both models are trained once on FluSight data up to a strict vintage cutoff
(before the test season), then back-tested on the 2023-24 season. Results are
saved to ``evaluation/results_dl.csv`` and compared against the ARIMA baseline
from STEP 3 if available.
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
from features.versioned_store import VersionedStore  # noqa: E402
from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402

SIGNAL = "flu_hosp_admissions"
SOURCE = "flusight"
EVAL_LOCATIONS = ["US", "06", "36"]                 # national, California, New York
HORIZONS = (1, 2, 3, 4)
TRAIN_END = "2023-10-28"                             # strictly before first origin
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results_dl.csv")


def _all_flusight_locations(store: VersionedStore):
    sig = store.list_signals()
    return sig[(sig["source"] == SOURCE) & (sig["signal"] == SIGNAL)]["location"].tolist()


def run() -> pd.DataFrame:
    store = VersionedStore()
    train_locs = _all_flusight_locations(store)
    print(f"Training on {len(train_locs)} locations with vintage cutoff {TRAIN_END}")

    models = {
        "tft": TFTModel(epochs=40, seed=0),
        "patchtst": PatchTSTModel(epochs=40, seed=0),
    }

    frames = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(store, signal=SIGNAL, source=SOURCE,
                  locations=train_locs, train_end_date=TRAIN_END, verbose=True)
        print(f"[{name}] trained in {time.time() - t0:.1f}s")
        for loc in EVAL_LOCATIONS:
            res = rolling_origin_backtest(
                store, model, ORIGINS,
                signal=SIGNAL, location=loc, source=SOURCE, horizons=HORIZONS,
            )
            res.insert(0, "model", name)
            frames.append(res)
            print(f"  [{name}] {loc:>3s}: {len(res)} scored forecasts")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(results)} rows -> {OUT_PATH}")

    # ---- comparison table (mean WIS by model x horizon), incl. ARIMA if present
    pieces = [results[["model", "horizon", "wis"]]]
    base_path = os.path.join(os.path.dirname(__file__), "results_baselines.csv")
    if os.path.exists(base_path):
        base = pd.read_csv(base_path)
        pieces.append(base[["model", "horizon", "wis"]])
    allres = pd.concat(pieces, ignore_index=True)
    pivot = allres.groupby(["model", "horizon"])["wis"].mean().unstack("horizon").round(1)
    print("\n=== Mean WIS by model x horizon (lower is better) ===")
    print(pivot.to_string())

    print("\n=== DL model summaries ===")
    for name in models:
        print(f"\n{name}:")
        print(summarize(results[results["model"] == name], by=("horizon",))
              .round(3).to_string(index=False))
    return results


if __name__ == "__main__":
    run()

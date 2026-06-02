"""Train the MIST-Transformer and evaluate it against all baselines.

Trains once on the FluSight panel up to a strict vintage cutoff, back-tests on
the 2023-24 season, saves ``evaluation/results_mist.csv`` and prints a combined
leaderboard against the statistical, mechanistic, and DL baselines.
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
from models.mist_transformer import MISTModel  # noqa: E402

SIGNAL = "flu_hosp_admissions"
SOURCE = "flusight"
EVAL_LOCATIONS = ["US", "06", "36"]
HORIZONS = (1, 2, 3, 4)
TRAIN_END = "2023-10-28"
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results_mist.csv")


def _all_flusight_locations(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SOURCE) & (sig["signal"] == SIGNAL)]["location"].tolist()


def run():
    store = VersionedStore()
    panel = _all_flusight_locations(store)
    print(f"Panel: {len(panel)} locations | vintage cutoff {TRAIN_END}")

    model = MISTModel(epochs=80, seed=0)
    t0 = time.time()
    model.fit(store, signal=SIGNAL, source=SOURCE, locations=panel,
              train_end_date=TRAIN_END, verbose=True)
    print(f"MIST trained in {time.time() - t0:.1f}s | "
          f"gamma_t={model.net.raw_gamma_t.item():.3f} "
          f"gamma_s={model.net.raw_gamma_s.item():.3f}")

    frames = []
    for loc in EVAL_LOCATIONS:
        res = rolling_origin_backtest(
            store, model, ORIGINS,
            signal=SIGNAL, location=loc, source=SOURCE, horizons=HORIZONS,
        )
        res.insert(0, "model", "mist")
        frames.append(res)
        print(f"  [mist] {loc:>3s}: {len(res)} scored forecasts")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(results)} rows -> {OUT_PATH}")

    # --- combined leaderboard ----------------------------------------------
    here = os.path.dirname(__file__)
    pieces = [results[["model", "horizon", "wis", "cov_50", "cov_95"]]]
    for fname in ("results_baselines.csv", "results_dl.csv"):
        p = os.path.join(here, fname)
        if os.path.exists(p):
            pieces.append(pd.read_csv(p)[["model", "horizon", "wis", "cov_50", "cov_95"]])
    allres = pd.concat(pieces, ignore_index=True)

    print("\n=== Mean WIS by model x horizon (lower is better) ===")
    print(allres.groupby(["model", "horizon"])["wis"].mean().unstack("horizon")
          .round(1).to_string())

    print("\n=== Overall mean WIS (all horizons) ===")
    overall = allres.groupby("model")["wis"].mean().sort_values()
    print(overall.round(1).to_string())

    print("\n=== MIST summary by horizon ===")
    print(summarize(results, by=("horizon",)).round(3).to_string(index=False))
    return results


if __name__ == "__main__":
    run()

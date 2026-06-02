"""Run the rolling-origin back-tester on the two baselines and save results.

Evaluates the ARIMA (statistical) and SEIR (mechanistic) baselines on the
FluSight target across a few locations and the 2023-24 season, then writes the
per-(model, location, origin, horizon) results to
``evaluation/results_baselines.csv``.
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import rolling_origin_backtest, summarize  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402
from models.baseline_mechanistic import SEIRModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402

SIGNAL = "flu_hosp_admissions"
SOURCE = "flusight"
LOCATIONS = ["US", "06", "36"]            # national, California, New York
HORIZONS = (1, 2, 3, 4)
ORIGINS = pd.date_range("2023-11-04", "2024-04-27", freq="7D")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results_baselines.csv")


def run() -> pd.DataFrame:
    store = VersionedStore()
    models = {"arima_stat": ARIMAQuantileModel(), "seir_mechanistic": SEIRModel()}

    frames = []
    for model_name, model in models.items():
        for loc in LOCATIONS:
            res = rolling_origin_backtest(
                store, model, ORIGINS,
                signal=SIGNAL, location=loc, source=SOURCE, horizons=HORIZONS,
            )
            if res.empty:
                continue
            res.insert(0, "model", model_name)
            frames.append(res)
            print(f"{model_name:18s} {loc:>3s}: {len(res):3d} scored forecasts")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(results)} rows -> {OUT_PATH}")

    print("\n=== Mean WIS by model x horizon (lower is better) ===")
    pivot = (results.groupby(["model", "horizon"])["wis"].mean()
             .unstack("horizon").round(1))
    print(pivot.to_string())

    print("\n=== Overall summary by model ===")
    for model_name in models:
        sub = results[results["model"] == model_name]
        print(f"\n{model_name}:")
        print(summarize(sub, by=("horizon",)).round(3).to_string(index=False))

    return results


if __name__ == "__main__":
    run()

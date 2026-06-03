"""Phase 4.4 — per-season validation across 2022-23 and 2023-24.

Re-runs a lean model set (full MIST v2 stack + ARIMA / SEIR / TFT / PatchTST) on
each flu season separately, training strictly on data before each season's first
origin, to confirm the headline results are not specific to one season. Writes
``results/season_wis.csv`` (per-season, per-model WIS / coverage).
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
from features.versioned_store import VersionedStore  # noqa: E402
from models.analogue_blend import AnalogueBlendModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402
from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
SIG, SRC = "flu_hosp_admissions", "flusight"
HORIZONS = (1, 2, 3, 4)

SEASONS = {
    "2022-23": {"train_end": "2022-10-29", "origins": ("2022-11-05", "2023-04-29")},
    "2023-24": {"train_end": "2023-10-28", "origins": ("2023-11-04", "2024-04-27")},
    "2024-25": {"train_end": "2024-09-28", "origins": ("2024-10-05", "2025-05-31")},
}

# Per-forecast columns kept for the pooled 3-season DM test (run_analysis / dm_3seasons).
_RECORD_COLS = ["season", "model", "forecast_date", "reference_date", "location",
                "horizon", "y_true", "median", "wis", "mae"]


def _panel(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SRC) & (sig["signal"] == SIG)]["location"].tolist()


def _bt_all(store, model, name, panel, origins):
    frames = []
    for loc in panel:
        r = rolling_origin_backtest(store, model, origins, signal=SIG, location=loc,
                                    source=SRC, horizons=HORIZONS)
        if not r.empty:
            r.insert(0, "model", name)
            frames.append(r)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run():
    os.makedirs(RES, exist_ok=True)
    store = VersionedStore()
    panel = _panel(store)
    rows = []
    records = []                       # per-forecast frames, all seasons/models
    for season, cfg in SEASONS.items():
        te = cfg["train_end"]
        origins = pd.date_range(*cfg["origins"], freq="7D")
        print(f"\n=== Season {season} | train_end {te} | {len(origins)} origins ===")

        mist = MISTModelV2(epochs=80, seed=0, use_conformal=False,
                           use_blend=True, use_mechanistic=True)
        mist.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=te)
        tft = TFTModel(epochs=40, seed=0)
        tft.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=te)
        patchtst = PatchTSTModel(epochs=40, seed=0)
        patchtst.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=te)

        # SEIR is omitted here: it is ~13 min/season on the full panel and already
        # appears in the main leaderboard; the cross-season generalisation check
        # only needs the competitive models.
        models = {
            "mist_v2": OnlineConformalModel(AnalogueBlendModel(mist), store,
                                            signal=SIG, source=SRC),
            "arima": ARIMAQuantileModel(),
            "tft": tft, "patchtst": patchtst,
        }
        for name, model in models.items():
            t0 = time.time()
            res = _bt_all(store, model, name, panel, origins)
            if res.empty:
                continue
            rows.append({"season": season, "model": name, "n": len(res),
                         "wis": res["wis"].mean(), "mae": res["mae"].mean(),
                         "cov_50": res["cov_50"].mean(), "cov_95": res["cov_95"].mean()})
            res = res.copy()
            res.insert(0, "season", season)
            records.append(res[_RECORD_COLS])
            print(f"  {name:10s}: WIS={res['wis'].mean():7.1f} "
                  f"cov50={res['cov_50'].mean():.2f} ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "season_wis.csv"), index=False)

    rec = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=_RECORD_COLS)
    rec.to_csv(os.path.join(RES, "season_records.csv"), index=False)

    print("\n=== season_wis.csv ===")
    print(out.pivot(index="model", columns="season", values="wis").round(1).to_string())
    print(f"\nseason_records.csv: {len(rec)} per-forecast rows -> {RES}")
    return out


if __name__ == "__main__":
    run()

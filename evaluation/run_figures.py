"""Phase 5 — generate the three paper figures from the v2 results + a trained MIST."""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.figures_paper import (attention_map_figure,  # noqa: E402
                                      case_study_figure, main_result_figure)
from features.versioned_store import VersionedStore  # noqa: E402
from models.analogue_blend import AnalogueBlendModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES, TAB, FIG = (os.path.join(ROOT, d) for d in ("results", "tables", "figures"))
SIG, SRC = "flu_hosp_admissions", "flusight"
TRAIN_END = "2023-10-28"
ATTN_ORIGIN = "2023-12-09"      # early-season rising week
HUB = "06"                      # California (high-connectivity hub)


def _panel(store):
    sig = store.list_signals()
    return sig[(sig["source"] == SRC) & (sig["signal"] == SIG)]["location"].tolist()


def run():
    os.makedirs(FIG, exist_ok=True)
    store = VersionedStore()
    panel = _panel(store)

    # --- 5.1 main result (from saved tables) -------------------------------
    phase_perf = pd.read_csv(os.path.join(RES, "phase_performance_v2.csv"), index_col=0)
    dm_p = pd.read_csv(os.path.join(TAB, "dm_significance.csv"), index_col=0)
    main_result_figure(phase_perf, os.path.join(FIG, "main_result"), dm_p=dm_p)
    print("Saved main_result.pdf/.png")

    # --- train MIST for the model-based figures ----------------------------
    base = MISTModelV2(epochs=80, seed=0, use_conformal=False,
                       use_blend=True, use_mechanistic=True)
    base.fit(store, signal=SIG, source=SRC, locations=panel, train_end_date=TRAIN_END)

    # --- 5.2 attention vs mobility -----------------------------------------
    out = attention_map_figure(base, store, signal=SIG, source=SRC, focal_loc=HUB,
                               origin=ATTN_ORIGIN, out_path=os.path.join(FIG, "attention_map.pdf"))
    print(f"Saved attention_map.pdf" if out else "  (attention map unavailable)")

    # --- 5.3 case study ----------------------------------------------------
    allres = pd.read_csv(os.path.join(RES, "results_all_v2.csv"),
                         parse_dates=["forecast_date", "reference_date"])
    mist = OnlineConformalModel(AnalogueBlendModel(base), store, signal=SIG, source=SRC)
    arima = ARIMAQuantileModel()
    out = case_study_figure(allres, store, mist, arima, signal=SIG, source=SRC,
                            out_path=os.path.join(FIG, "case_study.pdf"))
    print(f"Saved case_study.pdf" if out else "  (case study unavailable)")


if __name__ == "__main__":
    run()

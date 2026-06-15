"""Adjudicate the falsifiable success criterion for the phase-stacked hybrid.

`docs/neurips_gap_analysis.md` commits to five conditions the hybrid must meet
for the "phase-aware contribution" claim to be *earned*; otherwise the paper
pivots to a benchmark/applied framing. This module is the single place that
computes the verdict from artifacts, so the reproducibility gate, the paper
numbers, and the docs all read the same truth. It writes
``results/hybrid_verdict.json``.

The proposed model is ``stack_phase_conformal``. Conditions (all on the
season-unweighted WIS unless noted):

1. ``c1_le_base``     — hybrid WIS <= every individual base model.
2. ``c2_le_ens``      — hybrid WIS <= equal-weight, median, and perf-weighted ensembles.
3. ``c3_phase_gain``  — phase gating helps: ``stack_phase`` <= ``stack_global`` AND <= ``ens_perf``.
4. ``c4_calibration`` — conformal moves cov-50 toward 0.5 vs the un-conformal stack,
   without a WIS regression beyond a small tolerance.
5. ``c5_significant`` — the hybrid's advantage over the strongest *non-hybrid*
   competitor survives the clustered bootstrap (CI upper bound < 0) and
   Holm-adjusted DM (p < 0.05).

``earned`` is the AND of all five. Honesty guardrail: a ``False`` here is the
finding, and the paper must reflect it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.bootstrap import block_bootstrap_ci, pairwise_dm_adjusted  # noqa: E402
from evaluation.ensembles import COMPONENTS, ENSEMBLE_NAMES, LONG_PATH, per_forecast_metrics  # noqa: E402
from evaluation.run_multiseason import assemble_all  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))

HYBRID = "stack_phase_conformal"
STACKS = ("stack_global", "stack_phase", "stack_phase_conformal")
WIS_TOL = 0.02  # allow <=2% WIS regression from the conformal layer (calibration trade-off)


def _unweighted_wis(metrics: pd.DataFrame) -> pd.Series:
    return (metrics.groupby(["model", "season"])["wis"].mean()
            .groupby("model").mean())


def _cov50(metrics: pd.DataFrame) -> pd.Series:
    return metrics.groupby("model")["cov_50"].mean()


def compute(path: str = LONG_PATH, n_boot: int = 1000, seed: int = 0) -> dict:
    metrics = per_forecast_metrics(assemble_all(path))
    wis = _unweighted_wis(metrics)
    cov = _cov50(metrics)
    models = set(metrics["model"].unique())

    base_models = [m for m in models if m not in ENSEMBLE_NAMES and m not in STACKS]
    hybrid_wis = float(wis[HYBRID])

    # C1 / C2
    base_min = float(wis[base_models].min())
    base_argmin = str(wis[base_models].idxmin())
    ens_present = [m for m in ENSEMBLE_NAMES if m in models]
    ens_min = float(wis[ens_present].min()) if ens_present else np.inf
    c1 = hybrid_wis <= base_min + 1e-9
    c2 = hybrid_wis <= ens_min + 1e-9

    # C3: phase gating earns its place.
    c3 = bool(wis.get("stack_phase", np.inf) <= wis.get("stack_global", np.inf) + 1e-9
              and wis.get("stack_phase", np.inf) <= wis.get("ens_perf", np.inf) + 1e-9)

    # C4: conformal improves cov-50 toward nominal without material WIS regression.
    dist_before = abs(float(cov.get("stack_phase", np.nan)) - 0.5)
    dist_after = abs(float(cov.get("stack_phase_conformal", np.nan)) - 0.5)
    wis_before = float(wis.get("stack_phase", np.nan))
    wis_regress = (hybrid_wis - wis_before) / wis_before if wis_before else np.inf
    c4 = bool(dist_after <= dist_before + 1e-9 and wis_regress <= WIS_TOL)

    # C5: significance vs the strongest non-hybrid competitor.
    competitors = [m for m in models if m not in STACKS]
    comp_best = str(wis[competitors].idxmin())
    ci = block_bootstrap_ci(metrics, focal=HYBRID, n_boot=n_boot, seed=seed)
    dm = pairwise_dm_adjusted(metrics)
    ci_row = ci[ci["vs"] == comp_best]
    ci_ok = bool(not ci_row.empty and ci_row["ci_high"].iloc[0] < 0)
    pair = dm[((dm["a"] == HYBRID) & (dm["b"] == comp_best)) |
              ((dm["a"] == comp_best) & (dm["b"] == HYBRID))]
    dm_ok = bool(not pair.empty and pair["p_holm"].iloc[0] < 0.05
                 and hybrid_wis < float(wis[comp_best]))
    c5 = ci_ok and dm_ok

    verdict = {
        "hybrid": HYBRID,
        "components": COMPONENTS,
        "hybrid_wis_unweighted": round(hybrid_wis, 3),
        "best_base_model": base_argmin, "best_base_wis": round(base_min, 3),
        "best_ensemble_wis": round(ens_min, 3),
        "strongest_competitor": comp_best,
        "strongest_competitor_wis": round(float(wis[comp_best]), 3),
        "cov50_stack_phase": round(float(cov.get("stack_phase", np.nan)), 3),
        "cov50_hybrid": round(float(cov.get("stack_phase_conformal", np.nan)), 3),
        "conditions": {
            "c1_le_base": bool(c1), "c2_le_ens": bool(c2), "c3_phase_gain": bool(c3),
            "c4_calibration": bool(c4), "c5_significant": bool(c5),
        },
        "earned": bool(c1 and c2 and c3 and c4 and c5),
    }
    return verdict


def run(path: str = LONG_PATH, n_boot: int = 1000, seed: int = 0) -> dict:
    os.makedirs(RES, exist_ok=True)
    v = compute(path, n_boot=n_boot, seed=seed)
    with open(os.path.join(RES, "hybrid_verdict.json"), "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v, indent=2))
    print("\nVERDICT:", "EARNED (hybrid contribution holds)" if v["earned"]
          else "NOT EARNED -> paper pivots to benchmark/applied framing")
    return v


if __name__ == "__main__":
    run()

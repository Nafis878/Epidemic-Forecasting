"""Adjudicate the falsifiable success criterion for RAF (main-track gate).

Per ``docs/neurips_gap_analysis.md`` and the approved plan, RAF earns a
methodological main-track claim **only** if, on the *genuine-vintage* paired
(common-mask) benchmark, all four conditions hold. This is the single place the
verdict is computed, so the reproducibility gate, the paper, and the docs read
the same truth. It writes ``results/raf_verdict.json``.

Conditions (season-unweighted WIS on the common mask unless noted):

1. ``c1_beats_field``  — RAF WIS is the outright minimum: strictly below every
   other model, including PatchTST/TFT and ``ens_trimmed`` (the FluSight Hub's
   deployed robust default).
2. ``c2_significant``  — RAF's edge over the strongest *external* competitor (best
   model that is neither RAF nor its ablation) **and** over ``ens_trimmed``
   survives the clustered block bootstrap (CI upper bound < 0) **and** Holm-adjusted
   DM (p < 0.05).
3. ``c3_backfill``     — RAF significantly beats its ``raf_noback`` ablation (same
   backbone, revision correction OFF). This attributes the win to the *novelty*
   (Stage-1 backfill correction), not merely to the network — the decisive test.
4. ``c4_calibration``  — RAF's 50% coverage is no worse than the ablation's toward
   nominal 0.5, and 95% coverage stays within a reasonable band.

``earned`` is the AND of all four. Honesty guardrail (user decision: "main-track
or bust"): a ``False`` here means the project is shelved, not reframed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.bootstrap import block_bootstrap_ci, pairwise_dm_adjusted  # noqa: E402
from evaluation.ensembles import ENSEMBLE_NAMES, LONG_PATH, per_forecast_metrics  # noqa: E402
from evaluation.run_multiseason import assemble_all, common_mask_metrics  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))

RAF = "raf"
ABLATION = "raf_noback"
HUB_DEFAULT = "ens_trimmed"
COV95_BAND = (0.90, 0.98)


def _default_path() -> str:
    """Prefer the genuine-vintage dump (the only reportable RAF surface) if present."""
    genuine = os.path.join(RES, "quantiles_long_genuine.parquet")
    return genuine if os.path.exists(genuine) else LONG_PATH


def _unweighted_wis(metrics: pd.DataFrame) -> pd.Series:
    return (metrics.groupby(["model", "season"])["wis"].mean()
            .groupby("model").mean())


def _significant(ci: pd.DataFrame, dm: pd.DataFrame, focal: str, target: str,
                 wis: pd.Series) -> dict:
    """RAF vs ``target``: clustered-bootstrap CI upper < 0 AND Holm-DM p<0.05 AND lower WIS."""
    ci_row = ci[ci["vs"] == target]
    ci_ok = bool(not ci_row.empty and ci_row["ci_high"].iloc[0] < 0)
    pair = dm[((dm["a"] == focal) & (dm["b"] == target)) |
              ((dm["a"] == target) & (dm["b"] == focal))]
    p_holm = float(pair["p_holm"].iloc[0]) if not pair.empty else np.nan
    dm_ok = bool(not pair.empty and p_holm < 0.05 and float(wis[focal]) < float(wis[target]))
    return {"target": target, "ci_high": None if ci_row.empty else round(float(ci_row["ci_high"].iloc[0]), 3),
            "p_holm": None if pair.empty else round(p_holm, 4),
            "significant": bool(ci_ok and dm_ok)}


def compute(path: str | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    path = path or _default_path()
    metrics = common_mask_metrics(per_forecast_metrics(assemble_all(path)))
    wis = _unweighted_wis(metrics)
    cov = metrics.groupby("model")["cov_50"].mean()
    cov95 = metrics.groupby("model")["cov_95"].mean()
    models = set(metrics["model"].unique())

    if RAF not in models:
        # Legacy dump predating RAF (e.g. --skip-dump over an old parquet): don't
        # crash the pipeline, report the honest not-earned verdict with a note.
        return {"model": RAF, "dump": os.path.basename(path),
                "genuine_dump": path.endswith("quantiles_long_genuine.parquet"),
                "note": f"'{RAF}' absent from this dump; re-run the dump to evaluate it.",
                "conditions": {"c1_beats_field": False, "c2_significant": False,
                               "c3_backfill": False, "c4_calibration": False},
                "earned": False}

    raf_wis = float(wis[RAF])
    others = [m for m in models if m != RAF]
    external = [m for m in others if m != ABLATION]           # field minus the ablation
    comp_best = str(wis[external].idxmin()) if external else None

    # C1: outright leader of the whole field.
    c1 = bool(raf_wis <= float(wis[others].min()) + 1e-9)

    # C2 + C3: significance (clustered bootstrap + Holm-DM), focal = RAF.
    ci = block_bootstrap_ci(metrics, focal=RAF, n_boot=n_boot, seed=seed)
    dm = pairwise_dm_adjusted(metrics)
    sig_targets = {}
    for tgt in [t for t in (comp_best, HUB_DEFAULT) if t and t in models]:
        sig_targets[tgt] = _significant(ci, dm, RAF, tgt, wis)
    c2 = bool(sig_targets and all(s["significant"] for s in sig_targets.values()))

    abl = _significant(ci, dm, RAF, ABLATION, wis) if ABLATION in models else {"significant": False}
    c3 = bool(abl["significant"])

    # C4: calibration not worse than the ablation toward 0.5, 95% in-band.
    d_raf = abs(float(cov.get(RAF, np.nan)) - 0.5)
    d_abl = abs(float(cov.get(ABLATION, np.nan)) - 0.5) if ABLATION in models else np.inf
    cov95_raf = float(cov95.get(RAF, np.nan))
    c4 = bool(d_raf <= d_abl + 1e-9 and COV95_BAND[0] <= cov95_raf <= COV95_BAND[1])

    verdict = {
        "model": RAF,
        "dump": os.path.basename(path),
        "genuine_dump": path.endswith("quantiles_long_genuine.parquet"),
        "raf_wis_unweighted": round(raf_wis, 3),
        "ablation_raf_noback_wis": round(float(wis[ABLATION]), 3) if ABLATION in models else None,
        "strongest_external_competitor": comp_best,
        "strongest_external_wis": round(float(wis[comp_best]), 3) if comp_best else None,
        "hub_default_wis": round(float(wis[HUB_DEFAULT]), 3) if HUB_DEFAULT in models else None,
        "cov50_raf": round(float(cov.get(RAF, np.nan)), 3),
        "cov50_raf_noback": round(float(cov.get(ABLATION, np.nan)), 3) if ABLATION in models else None,
        "cov95_raf": round(cov95_raf, 3),
        "significance": {"vs_field_and_hub": sig_targets, "vs_ablation": abl},
        "conditions": {
            "c1_beats_field": c1, "c2_significant": c2,
            "c3_backfill": c3, "c4_calibration": c4,
        },
        "earned": bool(c1 and c2 and c3 and c4),
    }
    return verdict


def run(path: str | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    os.makedirs(RES, exist_ok=True)
    v = compute(path, n_boot=n_boot, seed=seed)
    with open(os.path.join(RES, "raf_verdict.json"), "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v, indent=2))
    if not v["genuine_dump"]:
        print("\nWARNING: verdict computed on a NON-genuine dump — not reportable. "
              "Run the genuine multi-disease dump before trusting this.")
    print("\nVERDICT:", "EARNED (RAF main-track claim holds)" if v["earned"]
          else "NOT EARNED -> shelve per 'main-track or bust'")
    return v


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="quantiles_long parquet (default: genuine if present)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(path=args.path, n_boot=args.n_boot, seed=args.seed)

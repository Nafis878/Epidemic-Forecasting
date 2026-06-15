"""One-shot GPU training driver for the genuine-vintage, multi-disease benchmark.

Designed for Google Colab Pro (a GPU runtime), where the expensive step — fitting
MIST / TFT / PatchTST across flu + COVID-19 + RSV and two genuine seasons — runs
on CUDA. It is intentionally self-contained:

    1. ingest genuine issue-dated vintages for all three diseases from the Delphi
       Epidata NHSN weekly signals (full week range, so 2020-2024 backfill gives
       training history while the 2024-25/2025-26 targets stay genuinely vintage);
    2. prove the vintages are genuine (revision report);
    3. train the base models on the GPU and dump full per-quantile forecasts to
       ``results/quantiles_long_genuine.parquet`` (carries a ``disease`` column);
    4. zip the artifacts so you can download them and run the cheap post-processing
       (ensembles, hybrid, bootstrap, paper numbers) locally / in CI.

Usage on Colab (see docs/COLAB.md for the full cell sequence):

    !python scripts/colab_train.py                      # ingest + train (GPU auto)
    !python scripts/colab_train.py --skip-ingest        # reuse an uploaded store
    !python scripts/colab_train.py --profile quick      # fast smoke (NOT reportable)

The only GPU-bound work is step 3; steps 1-2 are light. A free ``DELPHI_API_KEY``
in the environment lifts API rate limits during ingestion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.config import load_experiment  # noqa: E402


def _genuine_cfg() -> dict:
    cfg = (load_experiment() or {}).get("genuine")
    if not cfg:
        raise SystemExit("configs/experiment.yaml is missing the 'genuine' block.")
    return cfg


def step_ingest(time_values: str | None) -> None:
    from ingestion.vintage_delphi import ingest_genuine_panel
    g = _genuine_cfg()
    tv = time_values or g.get("ingest_time_values", "202001-202622")
    print(f"\n[1/4] Ingesting genuine vintages (NHSN weekly, time_values={tv}) ...")
    counts = ingest_genuine_panel(store_dir=g["store_dir"], time_values=tv)
    print("  ingested rows per disease:", counts)


def step_report() -> None:
    from evaluation.vintage_report import run as vintage_run
    g = _genuine_cfg()
    print("\n[2/4] Vintage-authenticity report (must show genuine_vintage = True) ...")
    vintage_run(store_dir=g["store_dir"],
                out_path=os.path.join("results", "vintage_authenticity_genuine.csv"))


def step_train(profile: str, device: str) -> str:
    from evaluation.quantile_dump import GENUINE_OUT, dump_genuine_panel
    print(f"\n[3/4] Training base models (device={device}, profile={profile}) + dumping ...")
    dump_genuine_panel(profile=profile, device=device, out_path=GENUINE_OUT)
    return GENUINE_OUT


def step_zip(out_parquet: str) -> str:
    zip_path = os.path.join("results", "genuine_artifacts.zip")
    print(f"\n[4/4] Zipping artifacts -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir("results"):
            if f.startswith("quantiles_long_genuine") and f.endswith(".parquet"):
                z.write(os.path.join("results", f), f)
        for extra in ("vintage_authenticity_genuine.csv", "run_profile.txt"):
            p = os.path.join("results", extra)
            if os.path.exists(p):
                z.write(p, extra)
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["quick", "full"], default="full")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse an existing/uploaded data/store_genuine")
    ap.add_argument("--time-values", default=None,
                    help="override the ingest week range (default from config)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    try:
        import torch
        print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
              + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except Exception as e:
        print("torch not importable:", e)

    if not args.skip_ingest:
        step_ingest(args.time_values)
    else:
        print("\n[1/4] Skipping ingestion (reusing data/store_genuine).")
    step_report()
    out = step_train(args.profile, args.device)
    zip_path = step_zip(out)

    print(f"\nDONE in {time.time() - t0:.0f}s. Download: {zip_path}")
    print("Then locally:  unzip into results/ and run the cheap post-processing "
          "(ensembles + hybrid + bootstrap + paper numbers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

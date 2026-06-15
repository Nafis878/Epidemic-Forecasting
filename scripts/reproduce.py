"""One-command reproduction of the multi-season benchmark + hybrid.

Runs the whole pipeline in dependency order:

    1. dump per-quantile base forecasts        -> results/quantiles_long.parquet
    2. multi-season aggregation + stratify      -> results/season_leaderboard.csv, ...
    3. clustered bootstrap CIs + adjusted DM     -> tables/bootstrap_ci.csv, ...
    4. adjudicate the hybrid success criterion   -> results/hybrid_verdict.json
    5. (full only) regenerate paper numbers       -> paper/numbers.json + macros.tex
    6. reproducibility gate                       -> exits non-zero on any failure

Profiles
--------
``--quick`` (default): small panel / few origins / few epochs. Validates the
pipeline and the tests in minutes. **Its numbers are never reported.**
``--full``: the entire panel, every weekly origin, paper epochs. The only
profile that may regenerate paper numbers.

Examples
--------
    python scripts/reproduce.py --quick
    python scripts/reproduce.py --full
    python scripts/reproduce.py --full --skip-dump   # reuse an existing dump

Expected runtime
----------------
``--quick``: ~1-2 min. ``--full``: dominated by base-model training (MIST/TFT/
PatchTST over 53 locations x 3 seasons) — budget a few hours on CPU; steps 2-6
are minutes once the dump exists.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_const", dest="profile", const="quick")
    g.add_argument("--full", action="store_const", dest="profile", const="full")
    ap.add_argument("--skip-dump", action="store_true",
                    help="reuse an existing results/quantiles_long.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.set_defaults(profile="quick")
    args = ap.parse_args()
    profile = args.profile
    t0 = time.time()

    print(f"\n[reproduce] profile = {profile}"
          + ("  (NUMBERS NOT FOR PAPER)" if profile == "quick" else ""))

    # 1. dump base quantiles
    if not args.skip_dump:
        from evaluation.quantile_dump import dump
        print("\n[1/6] dumping base quantiles ...")
        dump(profile=profile)
    else:
        print("\n[1/6] skipping dump (reusing existing quantiles_long.parquet)")

    # 2. multi-season aggregation
    print("\n[2/6] multi-season aggregation ...")
    from evaluation.run_multiseason import run as run_ms
    run_ms()

    # 3. bootstrap + adjusted DM
    print("\n[3/6] bootstrap CIs + adjusted DM ...")
    from evaluation.bootstrap import run as run_boot
    run_boot(n_boot=args.n_boot)

    # 4. hybrid verdict
    print("\n[4/6] adjudicating hybrid success criterion ...")
    from evaluation.hybrid_verdict import run as run_verdict
    verdict = run_verdict(n_boot=args.n_boot)

    # 5. paper numbers (full only)
    if profile == "full":
        print("\n[5/6] regenerating paper numbers ...")
        from paper.extract_numbers import run as run_numbers
        run_numbers()
    else:
        print("\n[5/6] skipping paper-number regeneration (quick profile is not reportable)")

    # 6. reproducibility gate
    print("\n[6/6] reproducibility gate ...")
    from evaluation.assert_reproducibility import check
    ok = check()

    print(f"\n[reproduce] done in {time.time() - t0:.0f}s. "
          f"hybrid earned = {verdict['earned']}. gate = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Dump full per-quantile base-model forecasts to a single long artifact.

The headline evaluation (``run_seasons.py``) keeps only median/WIS/MAE per
forecast, which is enough for a leaderboard but **not** enough to build
ensembles or the phase-stacked hybrid — those need every base model's full
23-quantile vector at every (origin, location, horizon). This module re-runs the
same leakage-safe vintage loop and persists the complete quantile records to
``results/quantiles_long.parquet``:

    [season, model, forecast_date, location, horizon, reference_date,
     quantile, value, y_true, phase_origin]

``phase_origin`` is the **vintage** epidemic phase at the origin
(:func:`evaluation.phase_gate.vintage_phase`, trailing-only — no leakage); it is
the gating signal the hybrid is allowed to use. Scoring truth ``y_true`` is the
*final* observed value at the target week (same as the back-tester).

Base-model training is the only expensive step. Once this parquet exists,
ensembles and the hybrid are cheap post-processing over it, so real
(non-quick) hybrid numbers follow from a single full base run.

Profiles
--------
``quick`` (default here): a small panel / few origins / few epochs to validate
the pipeline end-to-end in minutes. **Never** a source of paper numbers.
``full``: the entire panel, every weekly origin, paper epochs. The only profile
whose artifacts may be reported.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import _FINAL_AS_OF  # noqa: E402
from evaluation.metrics import DEFAULT_QUANTILES  # noqa: E402
from evaluation.phase_gate import vintage_phase  # noqa: E402
from evaluation.run_seasons import SEASONS, SIG, SRC, HORIZONS  # noqa: E402
from features.versioned_store import VersionedStore  # noqa: E402
from models.analogue_blend import AnalogueBlendModel  # noqa: E402
from models.baseline import PersistenceQuantileModel, SeasonalNaiveQuantileModel  # noqa: E402
from models.baseline_ml import GBQuantileModel  # noqa: E402
from models.baseline_stat import ARIMAQuantileModel  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402
from models.patch_tst import PatchTSTModel  # noqa: E402
from models.tft import TFTModel  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT_PARQUET = os.path.join(RES, "quantiles_long.parquet")

# Profiles: (locations | None=all, origin_stride, epochs dict). Paper numbers
# may only come from "full".
PROFILES = {
    "quick": {"n_locations": 6, "origin_stride": 3,
              "epochs": {"mist": 5, "tft": 5, "patchtst": 5}},
    "full": {"n_locations": None, "origin_stride": 1,
             "epochs": {"mist": 80, "tft": 40, "patchtst": 40}},
}

_COLS = ["season", "model", "forecast_date", "location", "horizon",
         "reference_date", "quantile", "value", "y_true", "phase_origin"]


def _panel(store, signal=SIG, source=SRC):
    sig = store.list_signals()
    return sig[(sig["source"] == source) & (sig["signal"] == signal)]["location"].tolist()


def _final_truth_map(store, location, signal=SIG, source=SRC):
    truth = store.get_vintage(signal, location, _FINAL_AS_OF, source=source)
    return {pd.Timestamp(d): float(v)
            for d, v in zip(truth["reference_date"], truth["value"])}


def backtest_quantiles(store, model, forecast_dates, *, location,
                       signal=SIG, source=SRC,
                       horizons=HORIZONS, quantiles=DEFAULT_QUANTILES):
    """Run the vintage loop for one model/location, keeping full quantiles.

    Returns a list of record dicts (one per origin x horizon x quantile) that
    have a realised target. Mirrors ``rolling_origin_backtest`` but persists the
    quantile vector and the vintage-origin phase instead of only median/WIS.
    """
    quantiles = list(quantiles)
    truth_map = _final_truth_map(store, location, signal, source)
    records = []
    for fdate in forecast_dates:
        fdate = pd.Timestamp(fdate)
        history = store.get_vintage(signal, location, fdate, source=source)
        if history.empty:
            continue
        phase = vintage_phase(history)  # vintage-only -> no leakage
        preds = model.predict(history=history, forecast_date=fdate,
                              horizons=horizons, quantiles=quantiles)
        if preds is None or preds.empty:
            continue
        for h, grp in preds.groupby("horizon"):
            target_date = pd.Timestamp(grp["reference_date"].iloc[0])
            y = truth_map.get(target_date)
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue  # target not yet realised -> cannot score
            for q, val in zip(grp["quantile"].to_numpy(float), grp["value"].to_numpy(float)):
                records.append({
                    "forecast_date": fdate, "location": str(location),
                    "horizon": int(h), "reference_date": target_date,
                    "quantile": float(q), "value": float(val),
                    "y_true": float(y), "phase_origin": phase,
                })
    return records


def canonicalize_reference_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose target week disagrees with the model consensus.

    A few base models (notably MIST for the first ~10 origins of a season) report
    a stale ``reference_date`` for a given ``(forecast_date, horizon)``. Keying the
    ensemble/hybrid on ``reference_date`` would then split one forecast into two
    pivot cells (and the stale rows are scored against the wrong truth week). We
    keep only the **majority** target date per (season, forecast_date, location,
    horizon) — the value the consensus of models agree on — so every model is
    aligned and any mis-anchored rows are excluded rather than mis-scored.
    """
    if df.empty:
        return df
    key = ["season", "forecast_date", "location", "horizon"]
    canon = (df.groupby(key)["reference_date"]
             .agg(lambda s: s.mode().iloc[0]).rename("_canon"))
    merged = df.merge(canon, on=key, how="left")
    out = merged[merged["reference_date"] == merged["_canon"]].drop(columns="_canon")
    return out.reset_index(drop=True)


def _build_models(store, panel, train_end, epochs, signal=SIG, source=SRC, device="cpu"):
    """Fit the trainable base models once on the season's training cutoff.

    ``device`` ("cpu"/"cuda") is passed to the neural models so the heavy training
    can run on a Colab GPU; the classical baselines ignore it.
    """
    mist = MISTModelV2(epochs=epochs["mist"], seed=0, use_conformal=False,
                       use_blend=True, use_mechanistic=True, device=device)
    mist.fit(store, signal=signal, source=source, locations=panel, train_end_date=train_end)
    tft = TFTModel(epochs=epochs["tft"], seed=0, device=device)
    tft.fit(store, signal=signal, source=source, locations=panel, train_end_date=train_end)
    patchtst = PatchTSTModel(epochs=epochs["patchtst"], seed=0, device=device)
    patchtst.fit(store, signal=signal, source=source, locations=panel, train_end_date=train_end)
    ml = GBQuantileModel(seed=0)
    ml.fit(store, signal=signal, source=source, locations=panel, train_end_date=train_end)
    return {
        "mist_v2": OnlineConformalModel(AnalogueBlendModel(mist), store, signal=signal, source=source),
        "tft": tft, "patchtst": patchtst, "ml": ml,
        "arima": ARIMAQuantileModel(),
        "persistence": PersistenceQuantileModel(),
        "seasonal_naive": SeasonalNaiveQuantileModel(),
    }


def auto_device(device: str = "auto") -> str:
    """Resolve "auto" to "cuda" when a GPU is present (Colab Pro), else "cpu"."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def dump(profile: str = "quick", seasons=None, out_path: str = OUT_PARQUET, *,
         store_dir=None, signal: str = SIG, source: str = SRC,
         seasons_cfg: dict | None = None, device: str = "auto",
         disease_tag: str | None = None) -> pd.DataFrame:
    """Dump full per-quantile base forecasts.

    Parameters added for the genuine / multi-disease / Colab-GPU runs:
    ``store_dir`` (defaults to the standard store), ``signal``/``source`` and
    ``seasons_cfg`` (per-disease selectors and season windows), and ``device``
    ("auto" picks CUDA on Colab). ``disease_tag`` is recorded in a ``disease``
    column when given, so multiple diseases can share one long artifact.
    """
    from evaluation.config import get_profile
    cfg = get_profile(profile, fallback=PROFILES[profile])
    seasons_cfg = seasons_cfg or SEASONS
    seasons = seasons or list(seasons_cfg)
    dev = auto_device(device)
    os.makedirs(RES, exist_ok=True)
    store = VersionedStore(store_dir=store_dir) if store_dir else VersionedStore()
    full_panel = _panel(store, signal, source)
    panel = full_panel if cfg["n_locations"] is None else full_panel[: cfg["n_locations"]]
    print(f"[dump] device={dev} signal={signal} source={source} "
          f"store={store_dir or 'data/store'} panel={len(panel)}")

    all_records = []
    for season in seasons:
        scfg = seasons_cfg[season]
        te = scfg["train_end"]
        origins = pd.date_range(*scfg["origins"], freq="7D")[:: cfg["origin_stride"]]
        print(f"\n=== {season} | profile={profile} | train_end {te} | "
              f"{len(panel)} locs x {len(origins)} origins ===")
        models = _build_models(store, panel, te, cfg["epochs"], signal, source, dev)
        for name, model in models.items():
            t0 = time.time()
            recs = []
            for loc in panel:
                recs.extend(backtest_quantiles(store, model, origins, location=loc,
                                               signal=signal, source=source))
            for r in recs:
                r["season"] = season
                r["model"] = name
            all_records.extend(recs)
            n_fc = len({(r["location"], r["forecast_date"], r["horizon"]) for r in recs})
            print(f"  {name:14s}: {n_fc:5d} forecasts ({time.time() - t0:.0f}s)")

    df = pd.DataFrame.from_records(all_records, columns=_COLS) if all_records \
        else pd.DataFrame(columns=_COLS)
    df = canonicalize_reference_dates(df)   # align models on the consensus target week
    if disease_tag is not None:
        df.insert(0, "disease", disease_tag)
    df.to_parquet(out_path, index=False)
    # Record which profile produced this dump, so paper-number extraction can
    # refuse to emit reportable macros from a quick (validation-only) run.
    with open(os.path.join(RES, "run_profile.txt"), "w") as fh:
        fh.write(profile)
    print(f"\nquantiles_long: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['season'].nunique()} seasons -> {out_path}")
    return df


GENUINE_OUT = os.path.join(RES, "quantiles_long_genuine.parquet")


def dump_genuine_panel(profile: str = "full", device: str = "auto",
                       out_path: str = GENUINE_OUT,
                       diseases: list[str] | None = None) -> pd.DataFrame:
    """Dump all genuine-vintage diseases into one long artifact (Colab/GPU entry).

    Reads the ``genuine`` block of ``configs/experiment.yaml`` (store, source,
    per-disease signal, genuine seasons) and trains the base models per disease
    on ``device`` (use ``"auto"`` to grab a Colab GPU). Output carries a
    ``disease`` column so the downstream ensemble/hybrid/aggregation code can be
    run per disease and pooled.
    """
    from evaluation.config import load_experiment
    cfg = (load_experiment() or {}).get("genuine")
    if not cfg:
        raise RuntimeError("configs/experiment.yaml is missing the 'genuine' block")
    diseases = diseases or list(cfg["diseases"])
    frames = []
    for dis in diseases:
        sig = cfg["diseases"][dis]["signal"]
        per_path = out_path.replace(".parquet", f"_{dis}.parquet")
        print(f"\n########## genuine dump: {dis} ({sig}) ##########")
        df = dump(profile=profile, out_path=per_path, store_dir=cfg["store_dir"],
                  signal=sig, source=cfg["source"], seasons_cfg=cfg["seasons"],
                  device=device, disease_tag=dis)
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    alldf.to_parquet(out_path, index=False)
    print(f"\n=== genuine multi-disease dump: {len(alldf)} rows, "
          f"diseases={sorted(alldf['disease'].unique()) if len(alldf) else []} -> {out_path}")
    return alldf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--signal", default=SIG)
    ap.add_argument("--source", default=SRC)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--disease-tag", default=None)
    ap.add_argument("--genuine", action="store_true",
                    help="dump all genuine-vintage diseases from the config block")
    args = ap.parse_args()
    if args.genuine:
        dump_genuine_panel(profile=args.profile, device=args.device, out_path=args.out
                           if args.out != OUT_PARQUET else GENUINE_OUT)
    else:
        dump(profile=args.profile, seasons=args.seasons, out_path=args.out,
             store_dir=args.store_dir, signal=args.signal, source=args.source,
             device=args.device, disease_tag=args.disease_tag)

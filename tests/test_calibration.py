"""Phase 2 tests: conformal recalibration improves coverage, with no leakage.

Uses a small synthetic 3-location panel (same construction as ``test_mist``) so the
tests are fast and deterministic.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.backtester import rolling_origin_backtest  # noqa: E402
from evaluation.calibration import ece, reliability  # noqa: E402
from features.versioned_store import SCHEMA_COLUMNS, VersionedStore  # noqa: E402
from models.mist_v2 import MISTModelV2  # noqa: E402
from models.online_conformal import OnlineConformalModel  # noqa: E402

LAG = pd.Timedelta(days=4)
LOCS = ["US", "06", "36"]


def _panel_store(tmp_path, n_weeks=80):
    """Synthetic two-season weekly panel so a prior season exists for calibration."""
    dates = pd.date_range("2022-09-03", periods=n_weeks, freq="7D")
    rng = np.random.default_rng(5)
    rows = []
    for k, loc in enumerate(LOCS):
        # two bell-shaped waves so the calibration season resembles the test season
        t = np.linspace(0, 4 * np.pi, n_weeks)
        base = (200 + 50 * k) + (1200 + 300 * k) * (np.sin(t) ** 2)
        vals = np.clip(base + rng.normal(0, 60, n_weeks), 0, None)
        for d, v in zip(dates, vals):
            rows.append(("synthetic", "flu_hosp_admissions", loc, d, d + LAG, float(v)))
    s = VersionedStore(store_dir=tmp_path / "store")
    s.ingest(pd.DataFrame(rows, columns=SCHEMA_COLUMNS))
    return s, dates


def _fit(store, train_end):
    m = MISTModelV2(context_length=8, horizon=2, patch_sizes=(2, 4), d_model=16,
                    n_heads=2, epochs=12, batch_size=8, seed=0, use_conformal=False)
    m.fit(store, signal="flu_hosp_admissions", source="synthetic",
          locations=LOCS, train_end_date=train_end, verbose=False)
    return m


def test_aci_improves_or_matches_coverage(tmp_path):
    store, dates = _panel_store(tmp_path)
    base = _fit(store, dates[55])
    origins = list(dates[56:72])
    aci = OnlineConformalModel(base, store, signal="flu_hosp_admissions",
                               source="synthetic", eta=1.5)

    def cov95(model):
        frames = [rolling_origin_backtest(store, model, origins,
                                          signal="flu_hosp_admissions", location=loc,
                                          source="synthetic", horizons=(1, 2))
                  for loc in LOCS]
        r = pd.concat(frames, ignore_index=True)
        return r["cov_95"].mean()

    # ACI should not reduce 95% coverage relative to the raw model.
    assert cov95(aci) >= cov95(base) - 1e-9


def test_split_conformal_deltas_are_populated(tmp_path):
    store, dates = _panel_store(tmp_path)
    m = MISTModelV2(context_length=8, horizon=2, patch_sizes=(2, 4), d_model=16,
                    n_heads=2, epochs=8, batch_size=8, seed=0, use_conformal=True)
    m.fit(store, signal="flu_hosp_admissions", source="synthetic",
          locations=LOCS, train_end_date=dates[55], verbose=False)
    assert set(m.conformal_delta.keys()) == {1, 2}
    assert all(len(v) > 0 for v in m.conformal_delta.values())


def test_aci_no_leakage(tmp_path):
    """Future-issued data must not change an ACI forecast made earlier."""
    store, dates = _panel_store(tmp_path)
    base = _fit(store, dates[55])
    aci = OnlineConformalModel(base, store, signal="flu_hosp_admissions",
                               source="synthetic", eta=1.5)
    fd = dates[60] + LAG
    hist = store.get_vintage("flu_hosp_admissions", "US", fd, source="synthetic")
    base._cache.clear()
    before = aci.predict(hist, fd, horizons=(1, 2))["value"].to_numpy()

    future = []
    for k in range(1, 5):
        ref = fd + pd.Timedelta(weeks=k)
        for loc in LOCS:
            future.append(("synthetic", "flu_hosp_admissions", loc, ref,
                           fd + pd.Timedelta(days=90), 999999.0))
    store.ingest(pd.DataFrame(future, columns=SCHEMA_COLUMNS))

    base._cache.clear()
    aci2 = OnlineConformalModel(base, store, signal="flu_hosp_admissions",
                                source="synthetic", eta=1.5)
    after = aci2.predict(hist, fd, horizons=(1, 2))["value"].to_numpy()
    assert np.allclose(before, after), "future-issued data leaked into ACI forecast"

"""WP1: the Revision-Aware Forecaster (RAF).

Covers the contract the back-tester needs (a valid, monotone, non-negative
quantile frame), the two properties the novelty claim rests on — (i) the
corrector is the *identity* until trained, so ``backfill`` off is byte-identical
to the revision-blind backbone, and (ii) once trained on upward backfill it
pushes preliminary values *up* toward their finalised level — and that RAF slots
into the model interface (``fit``/``predict``) reading signal/location off the
vintage frame.
"""

import numpy as np
import pandas as pd

from evaluation.metrics import DEFAULT_QUANTILES
from features.versioned_store import VersionedStore
from models.raf import RAFModel, RevisionCorrector

# Preliminary issues under-report; the signal matures upward over 3 weeks.
_RATIO = {0: 0.5, 1: 0.75, 2: 0.9}          # lag>=3 -> 1.0 (final)
_CORRECT_LAG = 3


def _build_store(tmp_path, n_weeks=44, locations=("US", "CA")):
    """Synthetic genuine-vintage store: each week matures upward over 3 issues."""
    start = pd.Timestamp("2023-01-07")
    rows = []
    for loc in locations:
        for i in range(n_weeks):
            ref = start + pd.Timedelta(weeks=i)
            final = 100.0 + 60.0 * np.sin(i / 6.0) + 5.0 * (i % 3)   # seasonal-ish, positive
            for lag in range(0, _CORRECT_LAG + 1):
                ratio = _RATIO.get(lag, 1.0)
                rows.append({"source": "nhsn", "signal": "flu", "location": loc,
                             "reference_date": ref,
                             "issue_date": ref + pd.Timedelta(weeks=lag),
                             "value": final * ratio})
    store = VersionedStore(store_dir=str(tmp_path / "store"))
    store.ingest(pd.DataFrame(rows))
    # Origin: after the last reference week, so its recent tail is still immature.
    train_end = start + pd.Timedelta(weeks=n_weeks - 1)
    return store, train_end


def _fit(tmp_path, backfill: bool):
    store, train_end = _build_store(tmp_path)
    model = RAFModel(context_length=12, horizon=4, backfill=backfill,
                     correct_lag=_CORRECT_LAG, max_lag=6, epochs=3,
                     corrector_epochs=150, seed=0)
    model.fit(store, signal="flu", source="nhsn", locations=["US", "CA"],
              train_end_date=train_end)
    history = store.get_vintage("flu", "US", train_end, source="nhsn")
    return model, history, train_end


def test_predict_returns_valid_quantile_frame(tmp_path):
    model, history, train_end = _fit(tmp_path, backfill=True)
    preds = model.predict(history, train_end, horizons=(1, 2, 3, 4))
    assert set(preds["horizon"]) == {1, 2, 3, 4}
    assert len(preds) == 4 * len(DEFAULT_QUANTILES)
    for _, grp in preds.groupby("horizon"):
        vals = grp.sort_values("quantile")["value"].to_numpy()
        assert np.all(np.diff(vals) >= -1e-6)        # monotone non-decreasing quantiles
        assert np.all(vals >= 0.0)                    # non-negative counts


def test_untrained_corrector_is_identity(tmp_path):
    """Zero-init corrector must not move values: backfill on == off at init."""
    corr = RevisionCorrector(max_lag=6)
    partial = np.array([10.0, 250.0, 0.0])
    out = corr.correct(partial, lag=np.array([0, 1, 2]), phase_code=np.array([0, 1, 2]))
    assert np.allclose(out, partial, atol=1e-4)


def test_backfill_off_matches_blind_backbone(tmp_path):
    """With backfill disabled RAF must consume the raw vintage unchanged."""
    model, history, train_end = _fit(tmp_path, backfill=False)
    # The decoder sees exactly `history`; corrected-history path is skipped.
    direct = model._decoder.predict(history, train_end, horizons=(1, 2, 3, 4))
    raf = model.predict(history, train_end, horizons=(1, 2, 3, 4))
    pd.testing.assert_frame_equal(direct.reset_index(drop=True),
                                  raf.reset_index(drop=True))


def test_trained_corrector_pushes_preliminary_up(tmp_path):
    """After training on upward backfill, a lag-0 preliminary value is corrected up."""
    model, _, _ = _fit(tmp_path, backfill=True)
    partial = np.array([50.0])                        # a fresh, under-reported week
    corrected = model._corrector.correct(partial, lag=np.array([0]),
                                         phase_code=np.array([0]))
    # lag-0 truth is ~0.5 of final, so the learned correction should raise it clearly.
    assert corrected[0] > partial[0] * 1.3


def test_backfill_changes_the_forecast(tmp_path):
    """The corrected tail must actually reach the forecast (novelty is not a no-op)."""
    on, history, train_end = _fit(tmp_path, backfill=True)
    off, _, _ = _fit(tmp_path, backfill=False)
    p_on = on.predict(history, train_end, horizons=(1,)).sort_values("quantile")["value"].to_numpy()
    p_off = off.predict(history, train_end, horizons=(1,)).sort_values("quantile")["value"].to_numpy()
    assert not np.allclose(p_on, p_off)

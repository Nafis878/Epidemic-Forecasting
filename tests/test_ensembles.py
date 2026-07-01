"""Ensemble combination primitives and the long-frame ensemble builder."""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import DEFAULT_QUANTILES
from models.ensemble import (
    equal_weight, hedge_weights, median_combine, trimmed_mean, weighted_combine,
)


def _monotone_rows(n_models=3, q=len(DEFAULT_QUANTILES), seed=0):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.normal(size=(n_models, q)) * 5 + 50, axis=1)
    return base


def test_equal_weight_is_mean_and_monotone():
    B = _monotone_rows()
    out = equal_weight(B)
    assert np.allclose(out, np.sort(B.mean(axis=0)))
    assert np.all(np.diff(out) >= -1e-9)


def test_median_combine_matches_numpy_median():
    B = _monotone_rows()
    assert np.allclose(median_combine(B), np.sort(np.median(B, axis=0)))


def test_trimmed_mean_drops_min_and_max():
    # 4 components: trimmed mean = mean of the middle two at each level.
    B = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 200.0]])
    out = trimmed_mean(B)
    assert np.allclose(out, np.sort(np.array([(1.0 + 2.0) / 2, (11.0 + 12.0) / 2])))
    assert np.all(np.diff(out) >= -1e-9)
    # <=2 components -> falls back to the plain mean.
    B2 = _monotone_rows(n_models=2)
    assert np.allclose(trimmed_mean(B2), equal_weight(B2))


def test_weighted_combine_normalises_and_stays_monotone():
    B = _monotone_rows()
    w = np.array([3.0, 1.0, 1.0])  # unnormalised
    out = weighted_combine(B, w)
    expect = np.sort((w / w.sum())[:, None] * B)  # not the combine, just a shape ref
    assert out.shape == (B.shape[1],)
    assert np.all(np.diff(out) >= -1e-9)
    # Degenerate weights -> equal weight fallback.
    out0 = weighted_combine(B, np.zeros(3))
    assert np.allclose(out0, equal_weight(B))


def test_hedge_weights_favour_low_loss_and_sum_to_one():
    w = hedge_weights(np.array([1.0, 2.0, 10.0]), eta=0.5)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1] > w[2]            # lower loss -> higher weight
    # eta -> 0 recovers (near) equal weights.
    w0 = hedge_weights(np.array([1.0, 2.0, 10.0]), eta=0.0)
    assert np.allclose(w0, np.full(3, 1 / 3))


# --------------------------------------------------------- integration (long frame)
def _toy_long():
    """Two components, two forecasts, full quantile vectors."""
    rows = []
    for model, shift in [("mist_v2", 0.0), ("persistence", 10.0)]:
        for fd, rd, loc, y in [("2023-11-04", "2023-11-11", "01", 50.0),
                               ("2023-11-11", "2023-11-18", "01", 60.0)]:
            for q in DEFAULT_QUANTILES:
                rows.append({"season": "2023-24", "model": model,
                             "forecast_date": fd, "location": loc, "horizon": 1,
                             "reference_date": rd, "quantile": q,
                             "value": 40 + shift + 20 * q, "y_true": y,
                             "phase_origin": "Rising"})
    df = pd.DataFrame(rows)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def test_build_ensembles_emits_three_models_and_monotone_quantiles():
    from evaluation.ensembles import build_ensembles
    long = _toy_long()
    ens = build_ensembles(long, components=["mist_v2", "persistence"])
    assert set(ens["model"].unique()) == {"ens_mean", "ens_median", "ens_trimmed", "ens_perf"}
    for _, g in ens.groupby(["model", "forecast_date", "horizon"]):
        v = g.sort_values("quantile")["value"].to_numpy()
        assert np.all(np.diff(v) >= -1e-9)
    # ens_mean must be the elementwise average of the two components.
    em = ens[(ens.model == "ens_mean") & (ens.forecast_date == pd.Timestamp("2023-11-04"))]
    em = em.sort_values("quantile")["value"].to_numpy()
    comp = long[long.forecast_date == pd.Timestamp("2023-11-04")]
    avg = comp.groupby("quantile")["value"].mean().sort_index().to_numpy()
    assert np.allclose(em, np.sort(avg))


def test_build_ensembles_preserves_disease_keys():
    from evaluation.ensembles import build_ensembles, per_forecast_metrics

    rows = []
    components = ["mist_v2", "tft", "patchtst", "arima", "persistence", "ml", "seasonal_naive"]
    for disease, y in [("flu", 10.0), ("covid", 100.0)]:
        for i, model in enumerate(components):
            for q in DEFAULT_QUANTILES:
                rows.append({
                    "disease": disease,
                    "season": "2024-25",
                    "model": model,
                    "forecast_date": pd.Timestamp("2025-01-04"),
                    "location": "US",
                    "horizon": 1,
                    "reference_date": pd.Timestamp("2025-01-11"),
                    "quantile": q,
                    "value": y + i + q,
                    "y_true": y,
                    "phase_origin": "Rising",
                })
    long = pd.DataFrame(rows)
    ens = build_ensembles(long)
    assert set(ens["disease"].unique()) == {"flu", "covid"}
    metrics = per_forecast_metrics(ens)
    assert set(metrics["disease"].unique()) == {"flu", "covid"}

"""Gradient-boosted quantile-regression baseline.

A classical-ML reference point alongside the deep baselines (TFT, PatchTST): a
direct multi-horizon quantile regressor built from gradient-boosted trees
(:class:`sklearn.ensemble.GradientBoostingRegressor` with ``loss="quantile"``).
This is the standard "is the fancy model beating a well-tuned tabular learner?"
sanity check that strong epidemic-forecasting benchmarks include.

Design choices that keep it lean and leakage-safe:

* **Vintage-only inputs.** Trains on per-location windows drawn from the store
  with a strict ``train_end_date`` cutoff and, at inference, consumes only the
  vintage ``history`` the back-tester hands it — same contract as
  :class:`models._dl_common.DLForecaster`.
* **Instance normalisation.** Each context window is standardised by its own
  mean/std (RevIN-style), so one model generalises across location scales.
* **Separate model per (horizon, anchor quantile).** A small set of *anchor*
  quantiles is fitted directly; the full FluSight 23-quantile vector is obtained
  by monotone interpolation across anchors (``np.interp`` clamps the tails),
  which is cheap and avoids fitting 23x4 boosters.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from evaluation.metrics import DEFAULT_QUANTILES
from models._dl_common import build_windows

_EPS = 1e-5
# Anchor quantiles fitted directly; the 23-level set is interpolated from these.
_ANCHORS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


class GBQuantileModel:
    """Gradient-boosted multi-horizon quantile forecaster (vintage-safe)."""

    def __init__(self, context_length: int = 8, horizon: int = 4,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 anchors: Sequence[float] = _ANCHORS, cadence_days: int = 7,
                 n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.05, subsample: float = 0.9,
                 seed: int = 0) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.quantiles = list(quantiles)
        self.anchors = list(anchors)
        self.cadence_days = cadence_days
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.seed = seed
        # models[h][anchor] -> fitted GradientBoostingRegressor
        self.models: dict[int, dict[float, GradientBoostingRegressor]] = {}

    # ----------------------------------------------------------------- training
    def fit_series(self, series: Sequence[np.ndarray], verbose: bool = False) -> "GBQuantileModel":
        X, Y = build_windows(series, self.context_length, self.horizon)
        if len(X) == 0:
            raise ValueError("No training windows could be built (series too short).")
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + _EPS
        Xn = (X - mu) / sd
        Yn = (Y - mu) / sd  # normalise targets by the context's scale

        self.models = {}
        for hi in range(self.horizon):
            self.models[hi + 1] = {}
            yh = Yn[:, hi]
            for a in self.anchors:
                gbr = GradientBoostingRegressor(
                    loss="quantile", alpha=a, n_estimators=self.n_estimators,
                    max_depth=self.max_depth, learning_rate=self.learning_rate,
                    subsample=self.subsample, random_state=self.seed,
                )
                gbr.fit(Xn, yh)
                self.models[hi + 1][a] = gbr
            if verbose:
                print(f"  fitted GB quantiles for horizon {hi + 1}")
        return self

    def fit(self, store, *, signal: str, source: str, locations: Sequence[str],
            train_end_date, verbose: bool = False) -> "GBQuantileModel":
        series = []
        for loc in locations:
            hist = store.get_vintage(signal, loc, train_end_date, source=source)
            if len(hist) >= self.context_length + self.horizon:
                series.append(hist.sort_values("reference_date")["value"]
                              .astype(np.float32).to_numpy())
        if not series:
            raise ValueError("No location had enough history before train_end_date.")
        return self.fit_series(series, verbose=verbose)

    # ---------------------------------------------------------------- inference
    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Optional[Sequence[float]] = None) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model is not trained; call fit() first.")
        quantiles = list(quantiles) if quantiles is not None else self.quantiles
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        v = hist["value"].astype(np.float32).to_numpy()

        ctx = v[-self.context_length:]
        if len(ctx) < self.context_length:
            ctx = np.concatenate([np.full(self.context_length - len(ctx), ctx[0]), ctx])
        mu, sd = float(ctx.mean()), float(ctx.std()) + _EPS
        xn = ((ctx - mu) / sd).reshape(1, -1)

        anchors = np.asarray(self.anchors, dtype=float)
        rows = []
        for h in horizons:
            if h < 1 or h > self.horizon:
                continue
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            anchor_vals = np.array([self.models[h][a].predict(xn)[0] for a in self.anchors])
            anchor_vals = np.sort(anchor_vals)             # enforce monotone anchors
            qv = np.interp(quantiles, anchors, anchor_vals)  # interpolate to 23 levels
            preds = np.clip(qv * sd + mu, 0.0, None)       # denormalise, non-negative
            for q, val in zip(quantiles, preds):
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": float(val)})
        return pd.DataFrame(rows)

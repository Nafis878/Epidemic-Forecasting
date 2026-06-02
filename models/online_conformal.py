"""Adaptive Conformal Inference (ACI) wrapper for online interval recalibration.

Split conformal assumes the calibration and test data are exchangeable. For
seasonal epidemic forecasting that fails: a calm off-season (or even a milder
prior season) does not represent the wave being forecast, so a *fixed* widening
mis-covers. **Adaptive Conformal Inference** (Gibbs & Candes, 2021, "Adaptive
Conformal Inference Under Distribution Shift") instead adjusts the interval width
*online* from realised coverage feedback and provably drives empirical coverage to
the nominal level regardless of distribution shift.

This wrapper is model-agnostic: it takes any base forecaster (here ``MISTModelV2``)
and, walking forecast origins in time order, maintains a per-interval-level width
multiplier. Before forecasting origin ``t`` it updates the multipliers using every
past forecast whose target has become **observable as of ``t``** (queried through
``get_vintage`` — strictly no leakage), then widens/tightens the base quantiles.

Because epidemic miscoverage is strongly *one-sided* (the median under-predicts
rising waves, so the truth exceeds the upper bound far more often than it falls
below the lower bound), the two tails are adapted **independently**::

    g_hi <- g_hi + eta * (I[y > upper] - alpha/2)
    g_lo <- g_lo + eta * (I[y < lower] - alpha/2)
    upper = median + exp(g_hi) * (q_hi - median)
    lower = median - exp(g_lo) * (median - q_lo)

At equilibrium each tail probability equals ``alpha/2``, i.e. central coverage
``1 - alpha``. One-sided adaptation widens only the tail that is missing, so
intervals stay as tight as the data allow (better WIS than symmetric widening).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import DEFAULT_QUANTILES


class OnlineConformalModel:
    """Stateful ACI wrapper around a base quantile forecaster.

    State is keyed by location and reset automatically when the forecast location
    changes, so a back-tester that loops ``for location: for origin:`` works
    unchanged.
    """

    def __init__(self, base, store, *, signal: str, source: str,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 eta: float = 1.5) -> None:
        self.base = base
        self.store = store
        self.signal = signal
        self.source = source
        self.quantiles = list(quantiles)
        self.eta = eta
        self._alphas = sorted({round(2.0 * q, 6) for q in self.quantiles if q < 0.5 - 1e-9})
        self._reset(None)

    # ---------------------------------------------------------------- state
    def _reset(self, loc: Optional[str]) -> None:
        self._loc = loc
        self._glo = {a: 0.0 for a in self._alphas}    # log multipliers, lower tail
        self._ghi = {a: 0.0 for a in self._alphas}    # log multipliers, upper tail
        # pending forecasts awaiting their outcome: target_date -> {alpha:(lo,up)}
        self._pending: dict[pd.Timestamp, dict] = {}
        self._scored: set = set()

    def _ingest_feedback(self, loc: str, as_of: pd.Timestamp) -> None:
        """Update multipliers from any pending target now observable as of ``as_of``."""
        if not self._pending:
            return
        obs = self.store.get_vintage(self.signal, loc, as_of, source=self.source)
        if obs.empty:
            return
        truth = {pd.Timestamp(d): float(v)
                 for d, v in zip(obs["reference_date"], obs["value"])}
        for tdate in sorted(self._pending):
            if tdate in self._scored or tdate not in truth:
                continue
            y = truth[tdate]
            for a, (lo, up) in self._pending[tdate].items():
                self._glo[a] += self.eta * ((1.0 if y < lo else 0.0) - a / 2.0)
                self._ghi[a] += self.eta * ((1.0 if y > up else 0.0) - a / 2.0)
            self._scored.add(tdate)

    # -------------------------------------------------------------- predict
    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        loc = str(history["location"].iloc[0]) if "location" in history.columns else None
        if loc != self._loc:
            self._reset(loc)
        fdate = pd.Timestamp(forecast_date)
        self._ingest_feedback(loc, fdate)

        preds = self.base.predict(history, fdate, horizons=horizons, quantiles=quantiles)
        if preds.empty:
            return preds

        q = np.asarray(self.quantiles)
        out = []
        for h, g in preds.groupby("horizon"):
            g = g.sort_values("quantile").copy()
            vals = g["value"].to_numpy(dtype=float)
            levels = g["quantile"].to_numpy(dtype=float)
            median = float(np.interp(0.5, levels, vals))
            new_vals = vals.copy()
            target_date = pd.Timestamp(g["reference_date"].iloc[0])
            rec = {}
            for j, ql in enumerate(levels):
                if abs(ql - 0.5) < 1e-9:
                    new_vals[j] = median
                    continue
                if ql < 0.5:                                   # lower tail
                    a = round(2.0 * ql, 6)
                    new_vals[j] = median - np.exp(self._glo.get(a, 0.0)) * (median - vals[j])
                    rec.setdefault(a, [None, None])[0] = new_vals[j]
                else:                                          # upper tail
                    a = round(2.0 * (1.0 - ql), 6)
                    new_vals[j] = median + np.exp(self._ghi.get(a, 0.0)) * (vals[j] - median)
                    rec.setdefault(a, [None, None])[1] = new_vals[j]
            new_vals = np.sort(np.clip(new_vals, 0.0, None))   # monotone, non-negative
            g["value"] = new_vals
            out.append(g)
            # store the widened intervals (only complete lo/up pairs) for feedback
            self._pending[target_date] = {
                a: (lo, up) for a, (lo, up) in rec.items()
                if lo is not None and up is not None}
        return pd.concat(out, ignore_index=True)

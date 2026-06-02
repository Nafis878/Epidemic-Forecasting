"""Analogue-prior wrapper (Phase 3.3): nudge rising-phase forecasts toward ILINet analogues.

A model-agnostic wrapper that, during the **rising phase**, shifts the base
model's quantile forecast toward the trajectory implied by the ``k`` most similar
historical ILINet rising windows (retrieved by DTW; see
:mod:`features.analogue`). The nudge is a soft prior — the whole quantile vector is
translated by ``w * (analogue_trajectory - base_median)`` per horizon, leaving the
spread (and hence the downstream conformal calibration) intact.

It composes with the rest of the stack as ``base -> AnalogueBlend -> ACI`` and is
trivially ablated by simply not wrapping. ILINet history is read from the final
``data/raw/ilinet.csv`` and mapped to each FluSight location through
:mod:`features.crosswalk`; analogues from the current/future season are excluded,
so there is no leakage.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from features.analogue import AnaloguePrior
from features.crosswalk import fips_to_ilinet

_ILINET_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ilinet.csv")


class AnalogueBlendModel:
    """Wrap a base forecaster with a rising-phase ILINet-analogue prior."""

    def __init__(self, base, *, window: int = 8, horizon: int = 4, k: int = 5,
                 weight: float = 0.3, ilinet_path: str = _ILINET_RAW) -> None:
        self.base = base
        self.window = window
        self.horizon = horizon
        self.k = k
        self.weight = weight
        self._priors: dict[str, Optional[AnaloguePrior]] = {}
        self._ilinet = self._load_ilinet(ilinet_path)

    @staticmethod
    def _load_ilinet(path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            return None
        # value: weighted ILI for national, unweighted otherwise.
        val = df["ili"].astype(float)
        if "wili" in df.columns:
            val = val.where(df["region"] != "nat", df["wili"].astype(float))
        from ingestion.ilinet import epiweek_to_saturday
        out = pd.DataFrame({
            "region": df["region"].astype(str),
            "reference_date": df["epiweek"].map(epiweek_to_saturday),
            "value": val,
        }).dropna(subset=["value", "reference_date"])
        return out

    def _prior_for(self, fips: str) -> Optional[AnaloguePrior]:
        if fips in self._priors:
            return self._priors[fips]
        prior = None
        region = fips_to_ilinet(fips)
        if self._ilinet is not None and region is not None:
            sub = self._ilinet[self._ilinet["region"] == region]
            if len(sub) >= self.window + self.horizon + self.k:
                prior = AnaloguePrior(sub, window=self.window, horizon=self.horizon)
        self._priors[fips] = prior
        return prior

    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        preds = self.base.predict(history, forecast_date, horizons=horizons,
                                  quantiles=quantiles)
        if preds.empty or self.weight <= 0:
            return preds
        loc = str(history["location"].iloc[0]) if "location" in history.columns else None
        prior = self._prior_for(loc) if loc is not None else None
        if prior is None:
            return preds

        recent = history.sort_values("reference_date")["value"].astype(float).to_numpy()
        if len(recent) < self.window:
            return preds
        # Only nudge while rising (recent upward slope) — analogues are rising windows.
        if recent[-1] <= recent[-min(4, len(recent))]:
            return preds
        traj = prior.forecast_values(recent, cutoff=pd.Timestamp(forecast_date), k=self.k)
        if traj is None:
            return preds

        out = []
        for h, g in preds.groupby("horizon"):
            g = g.sort_values("quantile").copy()
            vals = g["value"].to_numpy(dtype=float)
            levels = g["quantile"].to_numpy(dtype=float)
            median = float(np.interp(0.5, levels, vals))
            hi = int(h) - 1
            if 0 <= hi < len(traj):
                shift = self.weight * (float(traj[hi]) - median)
                vals = np.clip(np.sort(vals + shift), 0.0, None)
                g["value"] = vals
            out.append(g)
        return pd.concat(out, ignore_index=True)

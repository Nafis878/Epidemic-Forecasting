"""MIST-Transformer v2: calibrated, declining-phase-aware MIST.

``MISTModelV2`` subclasses the v1 :class:`~models.mist_transformer.MISTModel`
(kept intact as the clean ablation reference) and adds, as composable layers on
top of the trained network:

* **Split-conformal recalibration** (Phase 2.2). After training, a held-out
  time-ordered calibration split is used to widen/tighten each central
  prediction interval so empirical coverage matches the nominal level. We use the
  Conformalized-Quantile-Regression score (Romano et al. 2019; see also the split
  conformal tutorial of Angelopoulos & Bates 2022): for a target interval
  ``[q_lo, q_hi]`` the nonconformity is ``E = max(q_lo - y, y - q_hi)`` and the
  interval is adjusted by the ``ceil((n+1)(1-alpha))/n`` empirical quantile of
  ``E``. This is distribution-free and finite-sample valid.

Phase 3 adds R_t-conditioned output blending and the historical-analogue prior to
this same class.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import DEFAULT_QUANTILES
from models.mist_transformer import MISTModel, _EPS


class MISTModelV2(MISTModel):
    """MIST with split-conformal interval recalibration."""

    def __init__(self, *args, use_conformal: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_conformal = use_conformal
        # conformal_delta[horizon][alpha] = additive interval half-width adjustment.
        self.conformal_delta: dict[int, dict[float, float]] = {}
        # Symmetric central-interval alphas implied by the quantile set (<0.5 levels).
        self._alphas = sorted({round(2.0 * q, 6) for q in self.quantiles if q < 0.5 - 1e-9})

    # ------------------------------------------------------------------- fit
    def fit(self, store, *, signal: str, source: str, locations: Sequence[str],
            train_end_date, val_frac: float = 0.15, verbose: bool = False):
        super().fit(store, signal=signal, source=source, locations=locations,
                    train_end_date=train_end_date, val_frac=val_frac, verbose=verbose)
        if self.use_conformal:
            self._calibrate_conformal(train_end_date, val_frac)
            if verbose:
                d1 = {a: round(v, 1) for a, v in self.conformal_delta.get(1, {}).items()}
                print(f"  conformal interval adjustments @h=1 (alpha: half-width): {d1}")
        return self

    # --------------------------------------------------- conformal calibration
    @torch.no_grad()
    def _predict_windows(self, X: np.ndarray) -> np.ndarray:
        """Denormalised, quantile-sorted predictions for windows ``X`` (N,S,C)."""
        xt = torch.tensor(X, device=self.device)
        mu = xt.mean(-1, keepdim=True)
        sd = xt.std(-1, keepdim=True) + _EPS
        xn = (xt - mu) / sd
        preds = self.net(xn, xt, self.W)                       # (N,S,H,Q)
        preds = torch.sort(preds, dim=-1).values
        preds = (preds * sd.unsqueeze(-1) + mu.unsqueeze(-1)).clamp(min=0)
        return preds.cpu().numpy()

    def _calibrate_conformal(self, train_end_date, val_frac: float) -> None:
        """Compute per-(horizon, alpha) CQR adjustments in **normalised** units.

        The network is instance-normalised (RevIN), so a calibration split drawn
        from the calm off-season would yield tiny absolute adjustments that do not
        transfer to the high-volatility flu season. Two design choices address the
        distribution shift:

        * conformalise in the per-window **normalised** space (divide residuals by
          the context std) and rescale by the forecast window's own std at
          inference — a scale-adaptive, still distribution-free adjustment;
        * draw the calibration windows from a **prior flu season** (target month in
          Nov-Apr), which is far more exchangeable with the test season than the
          calm off-season tail. All such windows are pre-``train_end_date`` data, so
          this introduces no leakage.
        """
        df = self._panel_matrix(self.panel_locations, train_end_date)
        df = df.reindex(columns=self.panel_locations)
        mat = df.to_numpy(dtype=np.float32)
        dates = pd.DatetimeIndex(df.index)
        T, S = mat.shape
        C, H = self.context_length, self.horizon

        Xs, Ys, tgt_first = [], [], []
        for i in range(0, T - C - H + 1):
            Xs.append(mat[i:i + C].T)
            Ys.append(mat[i + C:i + C + H].T)
            tgt_first.append(dates[i + C])                      # date of the h=1 target
        if not Xs:
            return
        X, Y = np.stack(Xs), np.stack(Ys)                       # (N,S,C),(N,S,H)
        months = pd.DatetimeIndex(tgt_first).month
        in_season = np.isin(months, [11, 12, 1, 2, 3, 4])
        # Prefer a prior flu season for calibration; fall back to the tail.
        if in_season.sum() >= 8:
            idx = np.where(in_season)[0]
        else:
            n_val = max(1, int(len(X) * val_frac))
            idx = np.arange(len(X) - n_val, len(X))
        Xv, Yv = X[idx], Y[idx]
        preds = self._predict_windows(Xv)                       # (n,S,H,Q)

        # Per-window, per-location instance scale (mean/std over the context).
        mu = Xv.mean(axis=2, keepdims=True)                     # (n,S,1)
        sd = Xv.std(axis=2, keepdims=True) + _EPS               # (n,S,1)

        q = np.asarray(self.quantiles)
        self.conformal_delta = {}
        for hi in range(H):
            self.conformal_delta[hi + 1] = {}
            y_n = ((Yv[:, :, hi] - mu[:, :, 0]) / sd[:, :, 0]).reshape(-1)   # (n*S,)
            P = preds[:, :, hi, :]                              # (n,S,Q)
            P_n = ((P - mu) / sd).reshape(-1, len(q))           # normalised (mu,sd: n,S,1)
            n = len(y_n)
            for a in self._alphas:
                lo = np.array([np.interp(a / 2.0, q, row) for row in P_n])
                up = np.array([np.interp(1.0 - a / 2.0, q, row) for row in P_n])
                E = np.maximum(lo - y_n, y_n - up)              # normalised CQR score
                kq = min(1.0, np.ceil((n + 1) * (1.0 - a)) / n) if n > 0 else 1.0
                self.conformal_delta[hi + 1][a] = float(np.quantile(E, kq, method="higher"))

    # --------------------------------------------------------------- inference
    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        preds = super().predict(history, forecast_date, horizons=horizons,
                                quantiles=quantiles)
        if not self.use_conformal or preds.empty or not self.conformal_delta:
            return preds

        # Rescale the normalised adjustment by this forecast window's own std,
        # matching the instance-normalisation used during training.
        v = history.sort_values("reference_date")["value"].astype(float).to_numpy()
        ctx = v[-self.context_length:]
        if len(ctx) < self.context_length and len(ctx) > 0:
            ctx = np.concatenate([np.full(self.context_length - len(ctx), ctx[0]), ctx])
        scale = float(np.std(ctx)) + _EPS if len(ctx) else 1.0

        out = []
        for h, g in preds.groupby("horizon"):
            deltas = self.conformal_delta.get(int(h), {})
            g = g.copy()
            adj = []
            for q, val in zip(g["quantile"], g["value"]):
                if q < 0.5 - 1e-9:                              # lower bound -> widen down
                    val = val - deltas.get(round(2.0 * q, 6), 0.0) * scale
                elif q > 0.5 + 1e-9:                            # upper bound -> widen up
                    val = val + deltas.get(round(2.0 * (1.0 - q), 6), 0.0) * scale
                adj.append(max(0.0, val))
            g["value"] = np.sort(adj)                            # re-sort -> no crossing
            out.append(g)
        return pd.concat(out, ignore_index=True)

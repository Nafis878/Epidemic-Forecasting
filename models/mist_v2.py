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
from torch import nn

from evaluation.metrics import DEFAULT_QUANTILES
from models.mist_transformer import MISTNet, MISTModel, estimate_rt, _EPS


class MISTNetV2(MISTNet):
    """MIST network with an R_t-conditioned blend toward a persistence nowcast.

    Rationale (Phase 3.1): MIST excels in the rising phase but over-shoots in the
    declining phase, where a simple persistence/drift nowcast is hard to beat. We
    blend the two by the epidemic regime::

        alpha = sigmoid(beta * (R_t - 1))            # learnable scalar beta
        y_hat = alpha * MIST + (1 - alpha) * persistence

    so alpha -> 1 when R_t > 1 (rising: trust MIST) and alpha -> 0 when R_t < 1
    (declining: fall back to persistence). The persistence nowcast is a
    random-walk-with-drift in the network's normalised space (the normalised
    analogue of an AR(1)-in-log nowcast), and ``beta`` is trained end-to-end with
    the rest of the network through the pinball loss.
    """

    def __init__(self, *args, use_blend: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_blend = use_blend
        self.raw_beta = nn.Parameter(torch.tensor(3.0))      # init beta = 3.0

    def forward(self, x_norm, x_raw, W=None, return_gates: bool = False):
        out = super().forward(x_norm, x_raw, W, return_gates=return_gates)
        gates = None
        if return_gates:
            out, gates = out
        if self.use_blend:
            out = self._blend(out, x_norm, x_raw)
        return (out, gates) if return_gates else out

    def _blend(self, mist_out, x_norm, x_raw):
        # mist_out: (B,S,H,Q); x_norm,x_raw: (B,S,C)
        B, S, H, Q = mist_out.shape
        rt_last = estimate_rt(x_raw)[..., -1]                 # (B,S)
        alpha = torch.sigmoid(self.raw_beta * (rt_last - 1.0))    # (B,S)
        alpha = alpha.view(B, S, 1, 1)

        last = x_norm[..., -1]                                # (B,S)
        drift = (x_norm[..., -1] - x_norm[..., -4]) / 3.0     # per-step drift (normalised)
        steps = torch.arange(1, H + 1, device=mist_out.device, dtype=mist_out.dtype)
        pers = last.unsqueeze(-1) + drift.unsqueeze(-1) * steps   # (B,S,H)
        pers = pers.unsqueeze(-1)                             # (B,S,H,1) broadcast over Q
        return alpha * mist_out + (1.0 - alpha) * pers


class MISTModelV2(MISTModel):
    """MIST with R_t-conditioned blending (Phase 3.1) and conformal/ACI calibration."""

    def __init__(self, *args, use_conformal: bool = True, use_blend: bool = True,
                 mobility: str = "correlation", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_conformal = use_conformal
        self.use_blend = use_blend
        self.mobility = mobility
        # conformal_delta[horizon][alpha] = additive interval half-width adjustment.
        self.conformal_delta: dict[int, dict[float, float]] = {}
        # Symmetric central-interval alphas implied by the quantile set (<0.5 levels).
        self._alphas = sorted({round(2.0 * q, 6) for q in self.quantiles if q < 0.5 - 1e-9})

    def _compute_W(self, mat: np.ndarray) -> np.ndarray:
        """Real population x inverse-distance gravity matrix (Phase 4 / user decision).

        Falls back to the correlation proxy when ``mobility != 'gravity'`` or when no
        embedded geography covers the panel.
        """
        if self.mobility != "gravity":
            return super()._compute_W(mat)
        from features.mobility import gravity_matrix, has_geo
        locs = list(self.panel_locations)
        if not any(has_geo(l) for l in locs):
            return super()._compute_W(mat)
        W = gravity_matrix(locs)
        # For any location lacking geography, borrow its correlation row so it is
        # not spatially isolated.
        missing = [i for i, l in enumerate(locs) if not has_geo(l)]
        if missing:
            corr = super()._compute_W(mat)
            for i in missing:
                W[i, :] = corr[i, :]
                W[:, i] = corr[:, i]
            np.fill_diagonal(W, 1.0)
        return W

    def _build_net(self, context: int, horizon: int) -> MISTNetV2:
        return MISTNetV2(context, horizon, len(self.quantiles),
                         patch_sizes=self.patch_sizes, d_model=self.d_model,
                         n_heads=self.n_heads, n_experts=self.n_experts,
                         use_mechanistic=self.use_mechanistic, use_blend=self.use_blend)

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

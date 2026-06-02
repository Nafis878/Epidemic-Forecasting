"""Shared infrastructure for the deep-learning forecasters (TFT, PatchTST).

Both DL baselines share the same training/inference harness; only the network
architecture differs. This module provides:

* :func:`pinball_loss` — multi-quantile quantile (pinball) loss.
* :func:`build_windows` — sliding (context -> horizon) windows from per-location
  series.
* :class:`DLForecaster` — a trainer/predictor that wraps any quantile network.
  It trains once on data up to a cutoff and, at inference, consumes only the
  vintage ``history`` the back-tester hands it (so the no-leakage guarantee is
  preserved). Per-window **instance normalisation** (RevIN-style) is applied so
  the networks see scale-free inputs.

Network contract: a ``torch.nn.Module`` taking ``x`` of shape ``(B, context)``
and returning ``(B, horizon, n_quantiles)``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import DEFAULT_QUANTILES

_EPS = 1e-5


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss.

    preds: ``(B, H, Q)``; target: ``(B, H)``; quantiles: ``(Q,)``.
    """
    target = target.unsqueeze(-1)               # (B, H, 1)
    errors = target - preds                      # (B, H, Q)
    q = quantiles.view(1, 1, -1)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def build_windows(series: Sequence[np.ndarray], context: int, horizon: int):
    """Sliding windows from a list of 1-D series -> (X, Y) arrays.

    X: ``(N, context)``, Y: ``(N, horizon)``.
    """
    xs, ys = [], []
    for v in series:
        v = np.asarray(v, dtype=np.float32)
        n = len(v)
        for i in range(0, n - context - horizon + 1):
            xs.append(v[i:i + context])
            ys.append(v[i + context:i + context + horizon])
    if not xs:
        return np.empty((0, context), np.float32), np.empty((0, horizon), np.float32)
    return np.stack(xs), np.stack(ys)


def _instance_norm(x: np.ndarray):
    """Per-row mean/std. Returns (mu, sd) each shape (N, 1)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + _EPS
    return mu, sd


class DLForecaster:
    """Generic quantile DL forecaster: trains once, infers on vintage context."""

    def __init__(
        self,
        net_factory: Callable[[int, int, int], nn.Module],
        context_length: int = 16,
        horizon: int = 4,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        cadence_days: int = 7,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.net_factory = net_factory
        self.context_length = context_length
        self.horizon = horizon
        self.quantiles = list(quantiles)
        self.cadence_days = cadence_days
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.net: Optional[nn.Module] = None
        self._q = torch.tensor(self.quantiles, dtype=torch.float32, device=device)

    # ----------------------------------------------------------------- training
    def fit_series(self, series: Sequence[np.ndarray], verbose: bool = False) -> "DLForecaster":
        """Train on a collection of 1-D series (each a per-location history)."""
        set_seed(self.seed)
        X, Y = build_windows(series, self.context_length, self.horizon)
        if len(X) == 0:
            raise ValueError("No training windows could be built (series too short).")

        mu, sd = _instance_norm(X)
        Xn = (X - mu) / sd
        Yn = (Y - mu) / sd  # normalise target with the context's scale

        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net = self.net_factory(self.context_length, self.horizon, len(self.quantiles))
        self.net.to(self.device).train()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                preds = self.net(xb)                       # (B, H, Q)
                loss = pinball_loss(preds, yb, self._q)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:3d}  pinball={total / len(ds):.4f}")
        self.net.eval()
        return self

    def fit(self, store, *, signal: str, source: str, locations: Sequence[str],
            train_end_date, verbose: bool = False) -> "DLForecaster":
        """Build training series from the store using a strict vintage cutoff."""
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
    @torch.no_grad()
    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("Model is not trained; call fit() first.")
        hist = history.sort_values("reference_date")
        last_date = pd.Timestamp(hist["reference_date"].iloc[-1])
        v = hist["value"].astype(np.float32).to_numpy()

        # Take the last `context_length` values; left-pad if history is short.
        ctx = v[-self.context_length:]
        if len(ctx) < self.context_length:
            ctx = np.concatenate([np.full(self.context_length - len(ctx), ctx[0]), ctx])
        mu, sd = float(ctx.mean()), float(ctx.std()) + _EPS

        xn = torch.tensor((ctx - mu) / sd, dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        out = self.net(xn).squeeze(0).cpu().numpy()        # (H, Q)
        out = np.sort(out, axis=1)                          # enforce monotone quantiles
        preds = out * sd + mu
        preds = np.clip(preds, 0.0, None)                   # non-negative counts

        rows = []
        for h in horizons:
            if h < 1 or h > self.horizon:
                continue
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            for j, q in enumerate(self.quantiles):
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": float(preds[h - 1, j])})
        return pd.DataFrame(rows)

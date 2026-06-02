"""Temporal Fusion Transformer (TFT) quantile forecaster.

A compact, faithful TFT for the univariate weekly-admissions setting. PyTorch
Forecasting is not installed, so the architecture is implemented from scratch.
The key TFT building blocks are present:

* **Gated Residual Network (GRN)** with **GLU** gating — TFT's core nonlinearity.
* An **LSTM encoder** for locally-ordered temporal processing.
* **Interpretable multi-head self-attention** over the encoded sequence for
  long-range dependencies.
* **Quantile output heads** producing 23 quantiles per horizon.

(Static/known-future covariate variable-selection networks are omitted because
this baseline is univariate.) Training/inference, instance normalisation and the
pinball loss live in :mod:`models._dl_common`; this file defines the network and
a thin :class:`TFTModel` wrapper exposing the back-tester's model interface.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from evaluation.metrics import DEFAULT_QUANTILES
from models._dl_common import DLForecaster


class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021)."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc2(self.elu(self.fc1(x)))
        h = self.drop(h)
        return self.norm(x + self.glu(h))      # gated skip connection


class TFTNet(nn.Module):
    """Univariate TFT: embed -> LSTM -> GRN -> self-attention -> quantile heads."""

    def __init__(self, context: int, horizon: int, n_quantiles: int,
                 d_model: int = 32, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.n_quantiles = n_quantiles

        self.input_proj = nn.Linear(1, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_grn = GRN(d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_grn = GRN(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ELU(),
            nn.Linear(d_model, horizon * n_quantiles),
        )

    def forward(self, x):                       # x: (B, context)
        h = self.input_proj(x.unsqueeze(-1))    # (B, context, d_model)
        h, _ = self.lstm(h)
        h = self.post_lstm_grn(h)
        attn_out, _ = self.attn(h, h, h)        # interpretable self-attention
        h = self.attn_grn(h + attn_out)
        pooled = h[:, -1, :]                    # most-recent timestep summary
        out = self.head(pooled)                 # (B, horizon * Q)
        return out.view(-1, self.horizon, self.n_quantiles)


class TFTModel(DLForecaster):
    """Back-tester-facing TFT wrapper (see :class:`DLForecaster`)."""

    def __init__(self, context_length: int = 16, horizon: int = 4,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 d_model: int = 32, n_heads: int = 4, **kwargs) -> None:
        def factory(ctx, hor, nq):
            return TFTNet(ctx, hor, nq, d_model=d_model, n_heads=n_heads)

        super().__init__(factory, context_length=context_length, horizon=horizon,
                         quantiles=quantiles, **kwargs)

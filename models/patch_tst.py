"""PatchTST quantile forecaster (Nie et al., 2023, "A Time Series is Worth 64 Words").

Implemented from scratch in PyTorch. The defining ideas of PatchTST are present:

* **Patching** — the context window is split into overlapping patches; each patch
  is a "token", which slashes the attention sequence length and lets each token
  capture local sub-series semantics.
* **Channel independence** — trivially satisfied here (univariate series).
* A standard **Transformer encoder** over the patch tokens with learned
  positional embeddings.
* A **flatten + linear head** mapping encoded patches to ``horizon * n_quantiles``.

Instance normalisation (the RevIN idea) is handled by the shared
:class:`models._dl_common.DLForecaster` harness, which also owns training
(pinball loss) and inference. This file defines the network and a thin
:class:`PatchTSTModel` wrapper exposing the back-tester's model interface.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from evaluation.metrics import DEFAULT_QUANTILES
from models._dl_common import DLForecaster


class PatchTSTNet(nn.Module):
    """Patch -> embed -> Transformer encoder -> flatten -> quantile head."""

    def __init__(self, context: int, horizon: int, n_quantiles: int,
                 patch_len: int = 4, stride: int = 2, d_model: int = 48,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches over the context window (with end padding handled in forward).
        self.n_patches = max(1, (context - patch_len) // stride + 1)

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * d_model, horizon * n_quantiles),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, context) -> (B, n_patches, patch_len) via a sliding window."""
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches[:, : self.n_patches, :]

    def forward(self, x):                       # x: (B, context)
        patches = self._patchify(x)             # (B, n_patches, patch_len)
        tokens = self.patch_embed(patches) + self.pos_embed
        enc = self.norm(self.encoder(tokens))   # (B, n_patches, d_model)
        out = self.head(enc)                    # (B, horizon * Q)
        return out.view(-1, self.horizon, self.n_quantiles)


class PatchTSTModel(DLForecaster):
    """Back-tester-facing PatchTST wrapper (see :class:`DLForecaster`)."""

    def __init__(self, context_length: int = 16, horizon: int = 4,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 patch_len: int = 4, stride: int = 2, d_model: int = 48,
                 n_heads: int = 4, n_layers: int = 2, **kwargs) -> None:
        def factory(ctx, hor, nq):
            return PatchTSTNet(ctx, hor, nq, patch_len=patch_len, stride=stride,
                               d_model=d_model, n_heads=n_heads, n_layers=n_layers)

        super().__init__(factory, context_length=context_length, horizon=horizon,
                         quantiles=quantiles, **kwargs)

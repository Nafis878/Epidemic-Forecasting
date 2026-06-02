"""MIST-Transformer: Mechanistic-Informed Spatio-Temporal Transformer.

A novel quantile forecaster (implemented from scratch in PyTorch) that injects
epidemiological structure into a transformer. It operates on a *panel* of
locations jointly so that both temporal and spatial structure are modelled.
Three research components:

1. **Mechanistic Attention** (:class:`MechanisticAttention`). Standard scaled
   dot-product attention with an *additive mechanistic prior* added to the
   logits before softmax:

   * **Temporal prior** — from the estimated effective reproduction number
     ``R_t`` (renewal equation). Tokens in a similar reproduction regime are
     biased to attend to one another: ``M_temp[i,j] = -gamma_t * (R_i - R_j)^2``.
   * **Spatial prior** — from a **mobility-proximity** matrix ``W`` between
     locations: ``M_spat[a,b] = gamma_s * W[a,b]``. Real commuting/mobility
     flows would plug in here; absent that data we use a data-driven
     curve-connectivity proxy estimated on the *training* window only.

2. **Spatio-Temporal Mixture-of-Experts** (:class:`PhaseGatedMoE`). A set of
   experts whose gate is driven by **epidemic-phase features** (R_t, growth
   slope, acceleration), so different experts specialise to Rising / Peak /
   Declining regimes.

3. **Multi-Resolution Patching**. The context window is patched at several
   scales (default 2- and 4-week patches — the weekly analog of 7/14-day) and
   the resulting tokens are fused, capturing short- and longer-range dynamics.

Leakage safety: the panel context at every forecast origin is assembled through
:meth:`VersionedStore.get_vintage` with the forecast date as the cutoff, and the
mobility matrix is estimated only on pre-cutoff training data.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.metrics import DEFAULT_QUANTILES, weighted_interval_score
from models._dl_common import pinball_loss, set_seed

# Discretised generation-interval distribution (weeks) for the renewal equation.
GI_WEIGHTS = (0.6, 0.3, 0.1)
_EPS = 1e-5


# --------------------------------------------------------------------------- R_t
def estimate_rt(x: torch.Tensor, weights: Sequence[float] = GI_WEIGHTS) -> torch.Tensor:
    """Effective reproduction number via the renewal equation.

    ``R_t = I_t / sum_{s>=1} w_s * I_{t-s}``. ``x`` is ``(..., C)`` non-negative
    incidence; returns the same shape, clamped to a sensible range.
    """
    xc = x.clamp(min=0.0) + 1e-3
    denom = torch.zeros_like(xc)
    C = xc.shape[-1]
    for s, w in enumerate(weights, start=1):
        shifted = F.pad(xc, (s, 0), mode="replicate")[..., :C]
        denom = denom + w * shifted
    return (xc / denom).clamp(0.05, 10.0)


# --------------------------------------------------------------------- attention
class MechanisticAttention(nn.Module):
    """Multi-head attention with an additive mechanistic prior bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, d). bias: (T, T) or (B, T, T), added to attention logits.
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (B, h, T, dh)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if bias is not None:
            if bias.dim() == 2:
                bias = bias[None, None]                    # (1,1,T,T)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(1)                    # (B,1,T,T)
            scores = scores + bias
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class _Block(nn.Module):
    """Attention + FFN residual block with LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MechanisticAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, bias=None):
        x = self.norm1(x + self.attn(x, bias))
        x = self.norm2(x + self.ff(x))
        return x


# --------------------------------------------------------------------------- MoE
class PhaseGatedMoE(nn.Module):
    """Mixture-of-experts gated by epidemic-phase features.

    The gate consumes phase features (R_t, slope, acceleration, ...) so experts
    specialise to Rising / Peak / Declining regimes. Dense (soft) gating is used
    for stable training on limited epidemic data.
    """

    def __init__(self, d_model: int, n_experts: int = 3, n_phase_feats: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_phase_feats, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
            for _ in range(n_experts)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, phase_feats: torch.Tensor):
        # h: (B, S, d); phase_feats: (B, S, F)
        weights = torch.softmax(self.gate(phase_feats), dim=-1)   # (B, S, E)
        expert_out = torch.stack([e(h) for e in self.experts], dim=-2)  # (B,S,E,d)
        mixed = (weights.unsqueeze(-1) * expert_out).sum(dim=-2)   # (B, S, d)
        return self.norm(h + mixed), weights


# ----------------------------------------------------------------------- network
class MISTNet(nn.Module):
    """The MIST-Transformer network (panel in -> per-location quantiles out)."""

    def __init__(self, context: int, horizon: int, n_quantiles: int,
                 patch_sizes: Sequence[int] = (2, 4), d_model: int = 32,
                 n_heads: int = 2, n_experts: int = 3, dropout: float = 0.1,
                 use_mechanistic: bool = True) -> None:
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.patch_sizes = tuple(patch_sizes)
        # Ablation switch: when False, the mechanistic prior bias is removed from
        # both temporal and spatial attention (architecture otherwise identical).
        self.use_mechanistic = use_mechanistic

        # Multi-resolution patch embeddings (non-overlapping; stride = patch size).
        self.embeds = nn.ModuleList(nn.Linear(p, d_model) for p in self.patch_sizes)
        self.res_embed = nn.Parameter(torch.zeros(len(self.patch_sizes), d_model))
        self.tokens_per_res = [(context - p) // p + 1 for p in self.patch_sizes]
        self.n_tokens = sum(self.tokens_per_res)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.res_embed, std=0.02)

        # Learnable mechanistic-prior strengths (kept positive via softplus).
        self.raw_gamma_t = nn.Parameter(torch.tensor(0.0))
        self.raw_gamma_s = nn.Parameter(torch.tensor(0.0))

        self.temporal_block = _Block(d_model, n_heads, dropout)
        self.spatial_block = _Block(d_model, n_heads, dropout)
        self.moe = PhaseGatedMoE(d_model, n_experts=n_experts, n_phase_feats=4,
                                 dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, horizon * n_quantiles),
        )

    # --- feature helpers -----------------------------------------------------
    def _patch_tokens(self, x_norm, rt):
        """Build multi-resolution tokens and per-token R_t. Returns (tokens, r_tok)."""
        tok_list, r_list = [], []
        for i, p in enumerate(self.patch_sizes):
            xp = x_norm.unfold(-1, p, p)[:, :, : self.tokens_per_res[i], :]   # (B,S,nt,p)
            emb = self.embeds[i](xp) + self.res_embed[i]                      # (B,S,nt,d)
            tok_list.append(emb)
            rp = rt.unfold(-1, p, p)[:, :, : self.tokens_per_res[i], :].mean(-1)  # (B,S,nt)
            r_list.append(rp)
        tokens = torch.cat(tok_list, dim=2)              # (B,S,T,d)
        tokens = tokens + self.pos_embed[None]           # broadcast over S
        r_tok = torch.cat(r_list, dim=2)                 # (B,S,T)
        return tokens, r_tok

    @staticmethod
    def _phase_features(x_norm, rt):
        # x_norm,(B,S,C); rt (B,S,C) -> (B,S,4): [R_t_last, R_t_mean4, slope, accel]
        rt_last = rt[..., -1]
        rt_mean = rt[..., -4:].mean(-1)
        slope = x_norm[..., -1] - x_norm[..., -4]
        accel = x_norm[..., -1] - 2 * x_norm[..., -3] + x_norm[..., -5]
        return torch.stack([rt_last, rt_mean, slope, accel], dim=-1)

    def forward(self, x_norm: torch.Tensor, x_raw: torch.Tensor,
                W: Optional[torch.Tensor] = None, return_gates: bool = False):
        # x_norm, x_raw: (B, S, C);  W: (S, S)
        B, S, C = x_norm.shape
        rt = estimate_rt(x_raw)                           # (B,S,C)
        tokens, r_tok = self._patch_tokens(x_norm, rt)    # (B,S,T,d), (B,S,T)
        T = tokens.shape[2]
        d = tokens.shape[-1]

        # --- Temporal mechanistic attention (over tokens, per location) ---
        m_temp = None
        if self.use_mechanistic:
            gamma_t = F.softplus(self.raw_gamma_t)
            rdiff = r_tok.unsqueeze(-1) - r_tok.unsqueeze(-2)     # (B,S,T,T)
            m_temp = (-gamma_t * rdiff.pow(2)).reshape(B * S, T, T)
        h = tokens.reshape(B * S, T, d)
        h = self.temporal_block(h, m_temp).reshape(B, S, T, d)

        # --- Spatial mechanistic attention (over locations, per token) ---
        if S > 1:
            m_spat = None
            if self.use_mechanistic and W is not None:
                m_spat = F.softplus(self.raw_gamma_s) * W         # (S,S)
            hs = h.permute(0, 2, 1, 3).reshape(B * T, S, d)
            hs = self.spatial_block(hs, m_spat)
            h = hs.reshape(B, T, S, d).permute(0, 2, 1, 3)

        # --- Phase-gated MoE on the pooled per-location representation ---
        pooled = h.mean(dim=2)                                    # (B,S,d)
        phase = self._phase_features(x_norm, rt)                  # (B,S,4)
        mixed, gates = self.moe(pooled, phase)

        out = self.head(mixed).view(B, S, self.horizon, self.n_quantiles)
        if return_gates:
            return out, gates
        return out


# ------------------------------------------------------------------- forecaster
class MISTModel:
    """Back-tester-facing wrapper: panel training + leakage-safe vintage inference."""

    def __init__(self, context_length: int = 16, horizon: int = 4,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 patch_sizes: Sequence[int] = (2, 4), d_model: int = 32,
                 n_heads: int = 2, n_experts: int = 3, cadence_days: int = 7,
                 epochs: int = 80, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 16, seed: int = 0, device: str = "cpu",
                 use_mechanistic: bool = True) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.quantiles = list(quantiles)
        self.patch_sizes = tuple(patch_sizes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.cadence_days = cadence_days
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.use_mechanistic = use_mechanistic

        self.net: Optional[MISTNet] = None
        self.W: Optional[torch.Tensor] = None
        self.panel_locations: list = []
        self._store = None
        self._signal = None
        self._source = None
        self._cache: dict = {}
        self._q = torch.tensor(self.quantiles, dtype=torch.float32, device=device)

    # --- panel assembly ------------------------------------------------------
    def _panel_matrix(self, locations, as_of):
        """Rectangular (T, S) matrix of vintage values aligned on common dates."""
        cols = {}
        for loc in locations:
            h = self._store.get_vintage(self._signal, loc, as_of, source=self._source)
            if not h.empty:
                cols[loc] = h.set_index("reference_date")["value"].astype(float)
        df = pd.DataFrame(cols).dropna()
        return df

    # --- training ------------------------------------------------------------
    def fit(self, store, *, signal: str, source: str, locations: Sequence[str],
            train_end_date, val_frac: float = 0.15, verbose: bool = False):
        set_seed(self.seed)
        self._store, self._signal, self._source = store, signal, source
        self._cache.clear()

        df = self._panel_matrix(locations, train_end_date)
        self.panel_locations = list(df.columns)
        mat = df.to_numpy(dtype=np.float32)               # (T, S)
        T, S = mat.shape

        # Mobility-proximity proxy: training-window curve correlation (>=0).
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(mat.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        self.W = torch.tensor(np.clip(corr, 0.0, None), dtype=torch.float32,
                              device=self.device)

        C, H = self.context_length, self.horizon
        Xs, Ys = [], []
        for i in range(0, T - C - H + 1):
            Xs.append(mat[i:i + C].T)                      # (S, C)
            Ys.append(mat[i + C:i + C + H].T)              # (S, H)
        if not Xs:
            raise ValueError("Not enough panel history to build training windows.")
        X = np.stack(Xs)                                   # (N, S, C)
        Y = np.stack(Ys)                                   # (N, S, H)

        n_val = max(1, int(len(X) * val_frac))
        tr = slice(0, len(X) - n_val)
        va = slice(len(X) - n_val, len(X))

        self.net = MISTNet(C, H, len(self.quantiles), patch_sizes=self.patch_sizes,
                           d_model=self.d_model, n_heads=self.n_heads,
                           n_experts=self.n_experts,
                           use_mechanistic=self.use_mechanistic).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        Xt = torch.tensor(X, device=self.device)
        Yt = torch.tensor(Y, device=self.device)

        def norm(xb, yb):
            mu = xb.mean(-1, keepdim=True)
            sd = xb.std(-1, keepdim=True) + _EPS
            return (xb - mu) / sd, (yb - mu) / sd, mu, sd

        idx = np.arange(len(X))[tr]
        for epoch in range(self.epochs):
            self.net.train()
            np.random.shuffle(idx)
            total = 0.0
            for j in range(0, len(idx), self.batch_size):
                b = idx[j:j + self.batch_size]
                xb, yb = Xt[b], Yt[b]
                xn, yn, _, _ = norm(xb, yb)
                opt.zero_grad()
                preds = self.net(xn, xb, self.W)           # (B,S,H,Q)
                loss = pinball_loss(preds.reshape(-1, self.horizon, len(self.quantiles)),
                                    yn.reshape(-1, self.horizon), self._q)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            if verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                vw = self._val_wis(Xt[va], Yt[va], norm)
                print(f"  epoch {epoch:3d}  pinball={total / len(idx):.4f}  val_WIS={vw:.1f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def _val_wis(self, Xv, Yv, norm):
        if len(Xv) == 0:
            return float("nan")
        xn, _, mu, sd = norm(Xv, Yv)
        preds = self.net(xn, Xv, self.W)                   # (B,S,H,Q)
        preds = torch.sort(preds, dim=-1).values
        preds = preds * sd.unsqueeze(-1) + mu.unsqueeze(-1)
        preds = preds.clamp(min=0).cpu().numpy()
        y = Yv.cpu().numpy()
        B, S, H, Q = preds.shape
        flat_pred = preds.reshape(B * S * H, Q)
        flat_y = y.reshape(B * S * H)
        return float(np.mean(weighted_interval_score(flat_y, flat_pred, self.quantiles)))

    # --- inference -----------------------------------------------------------
    @torch.no_grad()
    def _forward_panel(self, as_of):
        df = self._panel_matrix(self.panel_locations, as_of)
        df = df.reindex(columns=self.panel_locations)
        mat = df.to_numpy(dtype=np.float32)                # (T, S)
        ctx = mat[-self.context_length:]
        if len(ctx) < self.context_length:                 # left-pad short history
            pad = np.repeat(ctx[:1], self.context_length - len(ctx), axis=0)
            ctx = np.concatenate([pad, ctx], axis=0)
        x = ctx.T[None]                                    # (1, S, C)
        xt = torch.tensor(x, device=self.device)
        mu = xt.mean(-1, keepdim=True)
        sd = xt.std(-1, keepdim=True) + _EPS
        xn = (xt - mu) / sd
        preds = self.net(xn, xt, self.W)                   # (1,S,H,Q)
        preds = torch.sort(preds, dim=-1).values
        preds = (preds * sd.unsqueeze(-1) + mu.unsqueeze(-1)).clamp(min=0)
        return preds.squeeze(0).cpu().numpy(), pd.Timestamp(df.index[-1])

    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("Model is not trained; call fit() first.")
        target_loc = str(history["location"].iloc[0]) if "location" in history.columns \
            else self.panel_locations[0]
        fd = pd.Timestamp(forecast_date)

        key = fd.isoformat()
        if key not in self._cache:
            self._cache[key] = self._forward_panel(fd)
        preds, last_date = self._cache[key]                # preds: (S,H,Q)

        if target_loc not in self.panel_locations:
            return pd.DataFrame(columns=["horizon", "reference_date", "quantile", "value"])
        s = self.panel_locations.index(target_loc)

        rows = []
        for h in horizons:
            if h < 1 or h > self.horizon:
                continue
            target = last_date + pd.Timedelta(days=self.cadence_days * h)
            for j, q in enumerate(self.quantiles):
                rows.append({"horizon": h, "reference_date": target,
                             "quantile": float(q), "value": float(preds[s, h - 1, j])})
        return pd.DataFrame(rows)

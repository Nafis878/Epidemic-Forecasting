"""RAF — a Revision-Aware Forecaster (NeurIPS main-track candidate).

Thesis: real-time epidemic forecasting is bottlenecked by *reporting revisions*.
The most recent weeks a forecaster sees are **preliminary** — NHSN hospital
admissions are revised (mostly upward) over subsequent issues — so a
revision-blind model (PatchTST/TFT/ARIMA) anchors each forecast on a biased,
under-reported tail. RAF is the first forecaster to take the **vintage triangle**
(``reference_week x issue_lag``; see :meth:`VersionedStore.get_triangle`) as a
first-class input and *de-bias the immature tail before forecasting*.

Two coupled stages, so the novelty is cleanly ablatable:

1. **Revision corrector** (:class:`RevisionCorrector`) — the novel module. From
   matured triangles it learns the backfill map ``E[log(final+1) | log(partial+1),
   issue_lag, phase]`` and, at inference, replaces each still-settling recent
   week's preliminary value with its revision-corrected estimate. Working in
   ``log1p`` space captures the multiplicative upward backfill without per-series
   normalisation, and it degrades gracefully to a no-op at large lag (matured
   weeks) — so with the module *off* the context is byte-identical to what the
   revision-blind baselines consume. This is the decisive ablation.
2. **Forecast decoder** — a standard quantile backbone (PatchTST-style) trained on
   finalised series via the shared :class:`~models._dl_common.DLForecaster`
   harness (pinball loss, RevIN-style instance norm). RAF keeps the backbone
   deliberately vanilla so any win is attributable to Stage 1, not a fancier net.

Leakage safety: RAF stashes the training ``store`` at ``fit`` time and, at
``predict`` time, re-queries ``get_triangle(signal, location, forecast_date)`` —
whose ``issue_date <= forecast_date`` guarantee is the same chokepoint the
back-tester relies on. ``signal``/``source``/``location`` are read off the
``history`` frame the back-tester passes, so RAF needs **no** back-tester change
and slots into ``quantile_dump._build_models`` like any other base model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation.metrics import DEFAULT_QUANTILES
from models._dl_common import DLForecaster, set_seed
from models.patch_tst import PatchTSTNet

_EPS = 1e-5
_PHASES = {"rising": 0, "peak": 1, "declining": 2, "unknown": 3}


def _phase_code(history) -> int:
    """Vintage phase -> integer code, tolerant of ``vintage_phase``'s capitalised
    labels and of empty/None history (both -> 'unknown')."""
    from evaluation.phase_gate import vintage_phase

    if history is None or len(history) == 0:
        return _PHASES["unknown"]
    return _PHASES.get(vintage_phase(history).lower(), _PHASES["unknown"])


# --------------------------------------------------------------- revision model
class RevisionCorrector(nn.Module):
    """Learns the backfill map ``log1p(final) ~ f(log1p(partial), lag, phase)``.

    Predicts the *log-ratio residual* ``r = log1p(final) - log1p(partial)`` from
    features ``[lag_scaled, log1p(partial), phase_onehot]``; the corrected value
    is ``expm1(log1p(partial) + r)``. Residual (rather than absolute) target means
    an untrained / zero-output net is the identity map, so the corrector can only
    help relative to the revision-blind baseline, never silently distort the
    signal it is uncertain about.
    """

    def __init__(self, hidden: int = 32, max_lag: int = 12) -> None:
        super().__init__()
        self.max_lag = max_lag
        in_dim = 2 + len(_PHASES)               # lag_scaled, logp_partial, phase one-hot
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start at the identity map (zero residual) for a safe, baseline-matching init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _features(self, lag, logp_partial, phase_code) -> torch.Tensor:
        lag = torch.as_tensor(lag, dtype=torch.float32).reshape(-1, 1)
        logp = torch.as_tensor(logp_partial, dtype=torch.float32).reshape(-1, 1)
        phase = torch.as_tensor(phase_code, dtype=torch.long).reshape(-1)
        onehot = torch.zeros(len(phase), len(_PHASES))
        onehot[torch.arange(len(phase)), phase.clamp(0, len(_PHASES) - 1)] = 1.0
        lag_scaled = (lag / float(self.max_lag)).clamp(0.0, 1.0)
        return torch.cat([lag_scaled, logp, onehot], dim=1)

    def forward(self, lag, logp_partial, phase_code) -> torch.Tensor:
        return self.net(self._features(lag, logp_partial, phase_code)).squeeze(-1)

    @torch.no_grad()
    def correct(self, partial: np.ndarray, lag: np.ndarray, phase_code: np.ndarray) -> np.ndarray:
        """Return revision-corrected values (non-negative) for preliminary cells."""
        logp = np.log1p(np.clip(np.asarray(partial, dtype=np.float64), 0.0, None))
        r = self.forward(lag, logp, phase_code).cpu().numpy().astype(np.float64)
        return np.clip(np.expm1(logp + r), 0.0, None)


def _train_corrector(corrector: RevisionCorrector, lag, logp_partial, resid, phase_code,
                     *, epochs: int, lr: float, seed: int) -> None:
    """Fit the corrector on matured (partial -> final) pairs via MSE on the log-residual."""
    if len(lag) == 0:
        return
    set_seed(seed)
    lag_t = torch.as_tensor(lag, dtype=torch.float32)
    logp_t = torch.as_tensor(logp_partial, dtype=torch.float32)
    y_t = torch.as_tensor(resid, dtype=torch.float32)
    ph_t = torch.as_tensor(phase_code, dtype=torch.long)
    opt = torch.optim.Adam(corrector.parameters(), lr=lr, weight_decay=1e-4)
    corrector.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = corrector(lag_t, logp_t, ph_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        opt.step()
    corrector.eval()


# ------------------------------------------------------------------- the model
class RAFModel:
    """Revision-aware quantile forecaster (back-tester model interface).

    Parameters
    ----------
    backfill:
        When ``True`` (default) Stage 1 corrects the immature context tail. When
        ``False`` RAF is byte-identical to its revision-blind backbone — the
        decisive ablation isolating the contribution.
    correct_lag:
        Only weeks whose latest issue lag is ``< correct_lag`` (i.e. still
        settling) are corrected; older weeks are treated as matured (no-op).
    """

    def __init__(self, context_length: int = 16, horizon: int = 4,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 backfill: bool = True, correct_lag: int = 8, max_lag: int = 12,
                 corrector_epochs: int = 200, corrector_lr: float = 5e-3,
                 epochs: int = 40, seed: int = 0, device: str = "cpu",
                 **backbone_kwargs) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.quantiles = list(quantiles)
        self.backfill = backfill
        self.correct_lag = correct_lag
        self.max_lag = max_lag
        self.corrector_epochs = corrector_epochs
        self.corrector_lr = corrector_lr
        self.seed = seed
        self.device = device

        def factory(ctx, hor, nq):
            return PatchTSTNet(ctx, hor, nq, **backbone_kwargs)

        self._decoder = DLForecaster(
            factory, context_length=context_length, horizon=horizon,
            quantiles=self.quantiles, epochs=epochs, seed=seed, device=device)
        self._corrector = RevisionCorrector(max_lag=max_lag)
        self._store = None
        self._signal: Optional[str] = None
        self._source: Optional[str] = None

    # ----------------------------------------------------------------- training
    def _corrector_pairs(self, store, signal, source, locations, train_end_date):
        """Matured (partial-at-lag, final) pairs from triangles issued <= cutoff.

        For each (location, reference_week) with a full-enough revision history,
        every early issue is a ``partial`` whose ``final`` is the latest matured
        issue (lag >= correct_lag) known by ``train_end_date`` — all leakage-safe.
        """
        from evaluation.phase_gate import vintage_phase

        lags, logp, resid, phase = [], [], [], []
        for loc in locations:
            tri = store.get_triangle(signal, loc, train_end_date, source=source)
            if tri.empty:
                continue
            collapsed = store.get_vintage(signal, loc, train_end_date, source=source)
            ph = _phase_code(collapsed if not collapsed.empty else None)
            for ref, grp in tri.groupby("reference_date"):
                grp = grp.sort_values("issue_lag_weeks")
                matured = grp[grp["issue_lag_weeks"] >= self.correct_lag]
                final = float((matured if not matured.empty else grp)["value"].iloc[-1])
                for _, row in grp.iterrows():
                    k = int(row["issue_lag_weeks"])
                    if k >= self.correct_lag:
                        continue                      # already matured -> nothing to learn
                    p = float(row["value"])
                    lp = np.log1p(max(p, 0.0))
                    lags.append(k); logp.append(lp)
                    resid.append(np.log1p(max(final, 0.0)) - lp); phase.append(ph)
        return (np.array(lags, dtype=np.float32), np.array(logp, dtype=np.float32),
                np.array(resid, dtype=np.float32), np.array(phase, dtype=np.int64))

    def fit(self, store, *, signal: str, source: str, locations: Sequence[str],
            train_end_date, verbose: bool = False) -> "RAFModel":
        self._store, self._signal, self._source = store, signal, source
        # Stage 2: finalised-series backbone (same training surface as the baselines).
        self._decoder.fit(store, signal=signal, source=source, locations=locations,
                          train_end_date=train_end_date, verbose=verbose)
        # Stage 1: the revision corrector (data-rich even when origins are thin).
        if self.backfill:
            lag, logp, resid, phase = self._corrector_pairs(
                store, signal, source, locations, train_end_date)
            _train_corrector(self._corrector, lag, logp, resid, phase,
                             epochs=self.corrector_epochs, lr=self.corrector_lr,
                             seed=self.seed)
            if verbose:
                print(f"  [RAF] corrector trained on {len(lag)} partial->final pairs")
        return self

    # ---------------------------------------------------------------- inference
    def _corrected_history(self, history: pd.DataFrame, forecast_date) -> pd.DataFrame:
        """Return ``history`` with its still-settling tail revision-corrected."""
        loc = str(history["location"].iloc[0])
        signal = str(history["signal"].iloc[0]) if "signal" in history else self._signal
        source = str(history["source"].iloc[0]) if "source" in history else self._source
        tri = self._store.get_triangle(signal, loc, forecast_date, source=source)
        if tri.empty:
            return history
        phase_code = _phase_code(history)

        hist = history.sort_values("reference_date").copy()
        vals = hist["value"].astype(np.float64).to_numpy()
        refs = pd.to_datetime(hist["reference_date"]).to_numpy()
        issues = pd.to_datetime(hist["issue_date"]).to_numpy() if "issue_date" in hist else refs
        for i in range(len(hist)):
            k = int((issues[i] - refs[i]) / np.timedelta64(7, "D"))
            if 0 <= k < self.correct_lag:            # still-settling recent week
                vals[i] = self._corrector.correct(
                    np.array([vals[i]]), np.array([k]), np.array([phase_code]))[0]
        hist["value"] = vals
        return hist

    def predict(self, history: pd.DataFrame, forecast_date,
                horizons: Sequence[int] = (1, 2, 3, 4),
                quantiles: Sequence[float] = None) -> pd.DataFrame:
        if history is None or history.empty:
            return pd.DataFrame(columns=["horizon", "reference_date", "quantile", "value"])
        hist = (self._corrected_history(history, forecast_date)
                if self.backfill and self._store is not None else history)
        return self._decoder.predict(hist, forecast_date, horizons=horizons,
                                     quantiles=quantiles)

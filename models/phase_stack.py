"""Phase-gated, online-learning, conformal stacking hybrid (the contribution).

Standalone MIST does not generalise across seasons (it wins only its development
season); no single base model dominates every regime. This module builds a
*hybrid* over the persisted base forecasts that routes weight to whichever models
are winning **in the current epidemic phase**, learns those weights online from
realised loss, and repairs interval calibration with a conformal layer — all
leakage-safe.

Three ideas, each isolable for ablation:

1. **Online Hedge weighting.** At origin ``t``, component ``m`` gets weight
   ``w_m ∝ exp(-eta · meanWIS_m)`` where ``meanWIS_m`` is taken over component
   forecasts whose target week was realised **strictly before ``t``**
   (``reference_date < t``) — prior seasons and earlier origins only. Equal
   weighting until enough history accrues.
2. **Phase gating.** The mean-loss pool is additionally restricted to the same
   *vintage* epidemic phase (``phase_origin`` from
   :func:`evaluation.phase_gate.vintage_phase`) and the same horizon, so the
   hybrid can prefer mechanistic models in the rising phase and others at the
   peak/decline. The ``global`` gate (horizon-only, no phase) is the ablation
   that isolates what phase-awareness actually buys.
3. **Conformalised quantile regression (CQR; Romano et al., 2019).** Per
   ``(horizon, alpha)`` the combined interval is widened/tightened by the
   ``(1-alpha)`` empirical quantile of strictly-prior nonconformity scores
   ``max(q_lo - y, y - q_hi)`` — driving central coverage (esp. the
   under-covered 50%) toward nominal without inflating every interval.

Outputs are long-format rows sharing the ``quantiles_long`` schema (``model`` =
the variant name), so they concatenate onto the base + ensemble frames for one
unified multi-season leaderboard. The ablation family this produces is::

    stack_global            (online Hedge, horizon-gated, no phase, no conformal)
    stack_phase             (+ phase gating)
    stack_phase_conformal   (+ CQR calibration)

paired with ``ens_mean`` / ``ens_median`` from :mod:`evaluation.ensembles` to
complete the five-way novelty isolation. The "phase-aware" claim is earned only
if ``stack_phase`` beats ``stack_global`` (and the perf-weighted ensemble) across
seasons; this code records the outcome either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.ensembles import COMPONENTS, ETA, _wide, key_cols, per_forecast_wis
from evaluation.metrics import DEFAULT_QUANTILES
from models.ensemble import hedge_weights, weighted_combine

MIN_HISTORY = 30          # min prior matched forecasts before trusting Hedge weights
_ALPHAS = tuple(sorted({round(2.0 * q, 6) for q in DEFAULT_QUANTILES if q < 0.5 - 1e-9}))


# ----------------------------------------------------------------- weighting
def _origin_weights(base_wis: pd.DataFrame, components, eta: float, by_phase: bool):
    """Precompute Hedge weights per gate key from strictly-prior matched loss.

    Returns ``dict`` keyed by ``(horizon, origin)`` (global gate) or
    ``(horizon, phase, origin)`` (phase gate) -> weight vector aligned to
    ``components``. Equal weights when the matched prior pool is too thin.
    """
    bw = base_wis.copy()
    bw["forecast_date"] = pd.to_datetime(bw["forecast_date"])
    bw["reference_date"] = pd.to_datetime(bw["reference_date"])
    n = len(components)
    eqw = np.full(n, 1.0 / n)
    out: dict = {}

    group_cols = (["disease"] if "disease" in bw.columns else [])
    group_cols += ["horizon", "phase_origin"] if by_phase else ["horizon"]
    for gkey, sub in bw.groupby(group_cols):
        origins = sorted(sub["forecast_date"].unique())
        ref = sub["reference_date"]
        for t in origins:
            prior = sub[ref < t]
            key = (*(gkey if isinstance(gkey, tuple) else (gkey,)), pd.Timestamp(t))
            if len(prior) < MIN_HISTORY:
                out[key] = eqw
                continue
            means = prior.groupby("model")["wis"].mean()
            losses = np.array([means.get(m, np.nan) for m in components])
            out[key] = eqw if np.isnan(losses).any() else hedge_weights(losses, eta)
    return out


# --------------------------------------------------------------- conformal
def _cqr_offsets(stack_wide: pd.DataFrame, levels: np.ndarray):
    """Per ``(horizon, origin, alpha)`` CQR offset from strictly-prior residuals.

    Calibration pool: prior realised hybrid forecasts of the **same horizon**
    (pooled over phases/locations for sample size). Offset is the ``(1-alpha)``
    empirical quantile of ``max(q_lo - y, y - q_hi)``.
    """
    idx = stack_wide.index.to_frame(index=False)
    y = idx["y_true"].to_numpy(dtype=float)
    fdates = pd.to_datetime(idx["forecast_date"]).to_numpy()
    rdates = pd.to_datetime(idx["reference_date"]).to_numpy()
    hor = idx["horizon"].to_numpy()
    diseases = idx["disease"].to_numpy() if "disease" in idx.columns else None
    M = stack_wide.to_numpy(dtype=float)

    lo_idx = {a: np.argmin(np.abs(levels - a / 2.0)) for a in _ALPHAS}
    hi_idx = {a: np.argmin(np.abs(levels - (1.0 - a / 2.0))) for a in _ALPHAS}

    offsets: dict = {}
    disease_values = np.unique(diseases) if diseases is not None else [None]
    for dis in disease_values:
        dm = np.ones(len(hor), dtype=bool) if diseases is None else (diseases == dis)
        for h in np.unique(hor[dm]):
            hm = dm & (hor == h)
            for t in np.unique(fdates[hm]):
                cal = hm & (rdates < t)
                key_base = ((str(dis), int(h), pd.Timestamp(t))
                            if diseases is not None else (int(h), pd.Timestamp(t)))
                if cal.sum() < MIN_HISTORY:
                    offsets[key_base] = {a: 0.0 for a in _ALPHAS}
                    continue
                ycal, Mcal = y[cal], M[cal]
                od = {}
                for a in _ALPHAS:
                    lo, hi = Mcal[:, lo_idx[a]], Mcal[:, hi_idx[a]]
                    scores = np.maximum(lo - ycal, ycal - hi)
                    lvl = min(1.0, np.ceil((len(scores) + 1) * (1 - a)) / len(scores))
                    od[a] = float(max(0.0, np.quantile(scores, lvl)))
                offsets[key_base] = od
    return offsets


def _apply_offsets(vec: np.ndarray, levels: np.ndarray, od: dict) -> np.ndarray:
    out = vec.copy()
    for j, q in enumerate(levels):
        if abs(q - 0.5) < 1e-9:
            continue
        a = round(2.0 * q, 6) if q < 0.5 else round(2.0 * (1.0 - q), 6)
        off = od.get(a, 0.0)
        out[j] = vec[j] - off if q < 0.5 else vec[j] + off
    return np.sort(np.clip(out, 0.0, None))


# ------------------------------------------------------------------- build
def build_stack(long: pd.DataFrame, *, components=COMPONENTS, eta: float = ETA,
                gate: str = "phase", conformal: bool = False,
                name: str | None = None) -> pd.DataFrame:
    """Build one hybrid variant as long-format rows.

    Parameters
    ----------
    gate:
        ``"phase"`` (horizon + vintage-phase gating) or ``"global"`` (horizon only).
    conformal:
        Apply the CQR calibration layer on top of the combined quantiles.
    name:
        ``model`` label for the output rows (defaults from gate/conformal).
    """
    if name is None:
        name = "stack_phase" if gate == "phase" else "stack_global"
        if conformal:
            name += "_conformal"

    base = long[long["model"].isin(components)].copy()
    if base.empty:
        return pd.DataFrame(columns=long.columns)
    base_wis = per_forecast_wis(base)
    weights = _origin_weights(base_wis, components, eta, by_phase=(gate == "phase"))

    w = _wide(base)
    levels = w.columns.to_numpy(dtype=float)
    eqw = np.full(len(components), 1.0 / len(components))
    keys = key_cols(base)

    # Pass 1: weighted combination per forecast.
    combined: dict[tuple, np.ndarray] = {}
    for key, sub in w.groupby(level=keys):
        mat = sub.droplevel(keys)
        present = [m for m in components if m in mat.index]
        B = mat.loc[present].to_numpy(dtype=float)
        keyt = key if isinstance(key, tuple) else (key,)
        keyd = dict(zip(keys, keyt))
        h, t, p = int(keyd["horizon"]), pd.Timestamp(keyd["forecast_date"]), keyd["phase_origin"]
        prefix = (keyd["disease"],) if "disease" in keyd else ()
        wkey = (*prefix, h, p, t) if gate == "phase" else (*prefix, h, t)
        wts = weights.get(wkey, eqw)
        if len(present) != len(components):
            wts = np.array([wts[components.index(m)] for m in present])
        combined[key] = weighted_combine(B, wts)

    stack_wide = pd.DataFrame.from_dict(combined, orient="index", columns=levels)
    stack_wide.index = pd.MultiIndex.from_tuples(combined.keys(), names=keys)

    # Pass 2 (optional): CQR calibration from strictly-prior residuals.
    if conformal:
        offsets = _cqr_offsets(stack_wide, levels)
        idx = stack_wide.index.to_frame(index=False)
        M = stack_wide.to_numpy(dtype=float)
        for i in range(len(M)):
            key_base = ((str(idx["disease"].iloc[i]), int(idx["horizon"].iloc[i]),
                         pd.Timestamp(idx["forecast_date"].iloc[i]))
                        if "disease" in idx.columns else
                        (int(idx["horizon"].iloc[i]), pd.Timestamp(idx["forecast_date"].iloc[i])))
            M[i] = _apply_offsets(M[i], levels, offsets.get(key_base, {}))
        stack_wide = pd.DataFrame(M, index=stack_wide.index, columns=levels)

    # Melt back to long format.
    rows = []
    for key, vec in zip(stack_wide.index, stack_wide.to_numpy(dtype=float)):
        keyd = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for q, val in zip(levels, vec):
            rows.append({**keyd, "model": name, "quantile": float(q), "value": float(val)})
    out = pd.DataFrame(rows)
    return out[long.columns] if not out.empty else out


def build_all_stacks(long: pd.DataFrame, *, components=COMPONENTS, eta: float = ETA) -> pd.DataFrame:
    """All three hybrid ablations as a single long frame."""
    frames = [
        build_stack(long, components=components, eta=eta, gate="global", conformal=False),
        build_stack(long, components=components, eta=eta, gate="phase", conformal=False),
        build_stack(long, components=components, eta=eta, gate="phase", conformal=True),
    ]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


if __name__ == "__main__":
    from evaluation.ensembles import load_long
    lg = load_long()
    st = build_all_stacks(lg)
    print(f"built hybrids {sorted(st['model'].unique())}: {len(st)} rows")

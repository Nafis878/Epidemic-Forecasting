"""Leakage-safe epidemic-phase gate for the phase-stacked hybrid.

The phase-gated stacker (``models.phase_stack``) routes base-model weights by the
*current* epidemic phase at each forecast origin. That gating signal must be
derivable from **vintage data only** (everything observable as of the origin),
otherwise the hybrid would leak the future.

:func:`vintage_phase` classifies the origin into ``Rising`` / ``Peak`` /
``Declining`` using a strictly *trailing* view of the vintage history — no
centered or forward-looking windows. This is deliberately distinct from
:func:`evaluation.visualizer.label_phases`, which labels the *target* week from
the **final** truth for retrospective stratification of results (an analysis
label, never a model input). Keeping the two separate makes the leakage boundary
explicit: ``vintage_phase`` is the only phase signal the hybrid is allowed to see.

Because a *trailing* view cannot peek at a still-higher future, the rule is
**slope-primary** rather than level-primary: a point in a monotonic rise (where
the current value is necessarily the trailing maximum) must still read as
``Rising``, not ``Peak``. We therefore classify by the recent normalised slope
and reserve ``Peak`` for a *flattening at a high level*::

    slope_norm > +tol  -> Rising
    slope_norm < -tol  -> Declining
    |slope_norm| <= tol-> Peak if level is near the trailing max, else Declining

This is the leakage-safe analogue of :func:`evaluation.visualizer.label_phases`
(which may use the final-truth, centered window for retrospective analysis).
"""

from __future__ import annotations

import pandas as pd

PHASES = ("Rising", "Peak", "Declining")


def vintage_phase(history: pd.DataFrame, *, peak_frac: float = 0.85,
                  smooth: int = 3, slope_window: int = 3, season_window: int = 21,
                  tol: float = 0.02) -> str:
    """Classify the forecast origin's epidemic phase from vintage history only.

    Parameters
    ----------
    history:
        Vintage truth as of the origin (columns ``reference_date``, ``value``),
        i.e. exactly what :meth:`VersionedStore.get_vintage` returns. Only the
        trailing portion is used, so the label cannot depend on the future.
    peak_frac:
        When the series has flattened, it is ``Peak`` if its level is within this
        fraction of the trailing-season maximum (else off-season -> ``Declining``).
    smooth:
        Trailing moving-average width (weeks) used to denoise the level/slope.
    slope_window:
        Lookback (weeks) over which the recent slope is measured.
    season_window:
        Width (weeks) of the trailing window defining the local seasonal maximum.
    tol:
        Dead-band on the per-week normalised slope separating flat from rising/falling.

    Returns
    -------
    str
        One of ``"Rising"``, ``"Peak"``, ``"Declining"``. Empty history -> ``"Rising"``.
    """
    if history is None or len(history) == 0:
        return "Rising"
    g = history.sort_values("reference_date")
    v = g["value"].astype(float).rolling(smooth, min_periods=1).mean()
    if len(v) < 2:
        return "Rising"
    last = float(v.iloc[-1])
    k = min(slope_window, len(v) - 1)
    recent_slope = (last - float(v.iloc[-1 - k])) / k          # avg change per week
    level = max(abs(last), 1.0)
    slope_norm = recent_slope / level

    if slope_norm > tol:
        return "Rising"
    if slope_norm < -tol:
        return "Declining"
    local_max = float(v.iloc[-season_window:].max())           # flattened: peak vs off-season
    return "Peak" if local_max > 0 and last >= peak_frac * local_max else "Declining"

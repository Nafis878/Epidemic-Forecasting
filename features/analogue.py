"""Historical-analogue retrieval from ILINet via Dynamic Time Warping (Phase 3.3).

ILINet covers ~25 flu seasons (1997-2025). For the current rising window we
retrieve the ``k`` most similar historical rising windows by **DTW distance on the
shape-normalised ILI%% signal**, then use the *mean continuation* of those analogues
as a soft prior on where the curve is heading. Shape normalisation (z-scoring each
window) lets us match the syndromic ILI%% signal to the hospital-admission target
despite their different units.

Leakage safety: the analogue library is built only from windows whose dates fall
**before the current season**, so a forecast never sees its own or future seasons.

The numpy DTW here avoids any extra dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_EPS = 1e-6


# --------------------------------------------------------------------------- DTW
def dtw_distance(a: np.ndarray, b: np.ndarray, band: Optional[int] = None) -> float:
    """Dynamic Time Warping distance between two 1-D sequences (Euclidean local cost).

    ``band`` optionally restricts the warping path to a Sakoe-Chiba band of the
    given width (in steps), which both speeds up the search and prevents
    pathological alignments.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = 1, m
        if band is not None:
            jlo = max(1, i - band)
            jhi = min(m, i + band)
        for j in range(jlo, jhi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def _znorm(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float)
    mu, sd = float(x.mean()), float(x.std()) + _EPS
    return (x - mu) / sd, mu, sd


# ------------------------------------------------------------------- library
class AnaloguePrior:
    """Build an analogue library from a long ILINet series and forecast from it.

    Parameters
    ----------
    series:
        Long DataFrame for one location with ``reference_date`` and ``value``
        (ILI%%), sorted ascending.
    window:
        Length of the matching window (weeks of recent history).
    horizon:
        Number of future weeks the analogue continuation provides.
    band:
        Sakoe-Chiba band for DTW.
    """

    def __init__(self, series: pd.DataFrame, window: int = 8, horizon: int = 4,
                 band: int = 3) -> None:
        self.window = window
        self.horizon = horizon
        self.band = band
        s = series.sort_values("reference_date")
        self._dates = pd.DatetimeIndex(s["reference_date"].to_numpy())
        self._vals = s["value"].astype(float).to_numpy()

    def _windows_before(self, cutoff: pd.Timestamp):
        """Yield (znorm_window, znorm_continuation, end_date) ending before cutoff."""
        cutoff = pd.Timestamp(cutoff)
        W, H = self.window, self.horizon
        out = []
        for i in range(len(self._vals) - W - H + 1):
            end_date = self._dates[i + W - 1]
            cont_end = self._dates[i + W + H - 1]
            # Exclude analogues that would peek at the current/future season.
            if cont_end >= cutoff:
                continue
            win = self._vals[i:i + W]
            cont = self._vals[i + W:i + W + H]
            zwin, mu, sd = _znorm(win)
            zcont = (cont - mu) / sd                          # continuation in window units
            out.append((zwin, zcont, end_date))
        return out

    def forecast_shape(self, recent: np.ndarray, cutoff, k: int = 5):
        """Mean z-normalised continuation of the ``k`` nearest historical analogues.

        ``recent`` is the recent (un-normalised) curve of the *target*; it is
        z-normalised before matching. Returns ``(shape, dists)`` where ``shape`` is
        a length-``horizon`` z-normalised forward trajectory, or ``None`` if the
        library is too small.
        """
        lib = self._windows_before(cutoff)
        if len(lib) < k:
            return None, None
        zq, _, _ = _znorm(np.asarray(recent, dtype=float)[-self.window:])
        dists = np.array([dtw_distance(zq, zwin, band=self.band) for zwin, _, _ in lib])
        idx = np.argsort(dists)[:k]
        shape = np.mean([lib[i][1] for i in idx], axis=0)
        return shape, dists[idx]

    def forecast_values(self, recent: np.ndarray, cutoff, k: int = 5):
        """Analogue forward trajectory mapped onto the target's own scale.

        The z-normalised analogue shape is rescaled by the target window's mean and
        std, giving an expected forward path in the target's units.
        """
        shape, _ = self.forecast_shape(recent, cutoff, k=k)
        if shape is None:
            return None
        r = np.asarray(recent, dtype=float)[-self.window:]
        mu, sd = float(r.mean()), float(r.std()) + _EPS
        return np.clip(mu + sd * shape, 0.0, None)

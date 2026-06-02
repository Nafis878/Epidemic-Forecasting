"""Gravity-model mobility matrix between US locations.

Replaces MIST's training-window *correlation* proxy with a real
**population x inverse-distance gravity** matrix, so the spatial mechanistic
attention reflects genuine population-flow structure and the "attention aligns
with mobility" analysis (Phases 4.3 / 5.2) is honest. No external download is
needed: state population (2020 census, in millions) and geographic centroid
(lat, lon) are static and embedded here.

Gravity flow between locations ``i`` and ``j``::

    W[i, j] ∝ pop_i * pop_j / dist(i, j) ** 2

with haversine distance. The matrix is symmetric, scaled to ``[0, 1]`` with unit
diagonal (matching the correlation proxy's conventions). National ``US`` is placed
at the conterminous-US centroid with population equal to the state total; any
location lacking centroid data (none by default) falls back gracefully.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# fips -> (lat, lon, population_millions). 50 states + DC + PR + US national.
GEO: dict[str, tuple[float, float, float]] = {
    "US": (39.83, -98.58, 331.4),
    "01": (32.81, -86.79, 5.02), "02": (61.37, -152.40, 0.73),
    "04": (33.73, -111.43, 7.15), "05": (34.97, -92.37, 3.01),
    "06": (36.12, -119.68, 39.54), "08": (39.06, -105.31, 5.77),
    "09": (41.60, -72.76, 3.61), "10": (39.32, -75.51, 0.99),
    "11": (38.90, -77.03, 0.69), "12": (27.77, -81.69, 21.54),
    "13": (33.04, -83.64, 10.71), "15": (21.09, -157.50, 1.46),
    "16": (44.24, -114.48, 1.84), "17": (40.35, -88.99, 12.81),
    "18": (39.85, -86.26, 6.79), "19": (42.01, -93.21, 3.19),
    "20": (38.53, -96.73, 2.94), "21": (37.67, -84.67, 4.51),
    "22": (31.17, -91.87, 4.66), "23": (44.69, -69.38, 1.36),
    "24": (39.06, -76.80, 6.18), "25": (42.23, -71.53, 7.03),
    "26": (43.33, -84.54, 10.08), "27": (45.69, -93.90, 5.71),
    "28": (32.74, -89.68, 2.96), "29": (38.46, -92.29, 6.15),
    "30": (46.92, -110.45, 1.08), "31": (41.13, -98.27, 1.96),
    "32": (38.31, -117.06, 3.10), "33": (43.45, -71.56, 1.38),
    "34": (40.30, -74.52, 9.29), "35": (34.84, -106.25, 2.12),
    "36": (42.17, -74.95, 20.20), "37": (35.63, -79.81, 10.44),
    "38": (47.53, -99.78, 0.78), "39": (40.39, -82.76, 11.80),
    "40": (35.57, -96.93, 3.96), "41": (44.57, -122.07, 4.24),
    "42": (40.59, -77.21, 13.00), "44": (41.68, -71.51, 1.10),
    "45": (33.86, -80.95, 5.12), "46": (44.30, -99.44, 0.89),
    "47": (35.75, -86.69, 6.92), "48": (31.05, -97.56, 29.15),
    "49": (40.15, -111.86, 3.27), "50": (44.05, -72.71, 0.64),
    "51": (37.77, -78.17, 8.63), "53": (47.40, -121.49, 7.71),
    "54": (38.49, -80.95, 1.79), "55": (44.27, -89.62, 5.89),
    "56": (42.76, -107.30, 0.58), "72": (18.22, -66.43, 3.29),
}

_EARTH_KM = 6371.0


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_KM * math.asin(math.sqrt(h))


def gravity_matrix(locations: Sequence[str], min_dist_km: float = 50.0) -> np.ndarray:
    """Symmetric gravity matrix over ``locations`` (FIPS), scaled to [0,1], unit diagonal.

    Locations without embedded geography contribute no off-diagonal weight (their
    row/column is zero except the diagonal), so the caller can fall back as needed.
    """
    n = len(locations)
    W = np.zeros((n, n), dtype=float)
    have = [loc in GEO for loc in locations]
    for i in range(n):
        if not have[i]:
            continue
        lat_i, lon_i, pop_i = GEO[locations[i]]
        for j in range(i + 1, n):
            if not have[j]:
                continue
            lat_j, lon_j, pop_j = GEO[locations[j]]
            d = max(_haversine((lat_i, lon_i), (lat_j, lon_j)), min_dist_km)
            flow = pop_i * pop_j / (d ** 2)
            W[i, j] = W[j, i] = flow
    mx = W.max()
    if mx > 0:
        W = W / mx
    np.fill_diagonal(W, 1.0)
    return W


def total_outflow(locations: Sequence[str]) -> dict[str, float]:
    """Per-location total gravity outflow (row sum minus the diagonal)."""
    W = gravity_matrix(locations)
    out = W.sum(axis=1) - np.diag(W)
    return {loc: float(v) for loc, v in zip(locations, out)}


def has_geo(loc: str) -> bool:
    return loc in GEO

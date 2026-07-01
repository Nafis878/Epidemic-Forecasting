"""WP3: the RAF main-track gate (evaluation.raf_verdict).

Checks the adjudication mechanics on constructed leaderboards: a legacy dump with
no RAF degrades to an honest not-earned verdict (never crashes the pipeline), and
a dump where RAF is the outright WIS leader sets ``c1_beats_field`` while the
significance/ablation conditions remain gated by the statistical tests.
"""

import numpy as np
import pandas as pd

from evaluation.ensembles import ENSEMBLE_COMPONENTS
from evaluation.metrics import DEFAULT_QUANTILES
from evaluation.raf_verdict import compute

_Q = list(DEFAULT_QUANTILES)


def _long(models_bias: dict, tmp_path) -> str:
    """Build a minimal quantiles_long parquet; each model is centered at
    ``y_true + bias`` so mean-WIS ordering is deterministic (bias 0 => best)."""
    seasons = {"2024-25": pd.Timestamp("2024-12-28"), "2025-26": pd.Timestamp("2025-10-04")}
    rows = []
    rng = np.random.default_rng(0)
    for season, f0 in seasons.items():
        for loc in ("US", "CA"):
            for w in range(8):                       # 8 origins per (season, location)
                fdate = f0 + pd.Timedelta(weeks=w)
                for h in (1, 2):
                    ref = fdate + pd.Timedelta(weeks=h)
                    y = 100.0 + 20.0 * rng.standard_normal()
                    for model, bias in models_bias.items():
                        for q in _Q:
                            val = max(0.0, y + bias + 40.0 * (q - 0.5))
                            rows.append({"season": season, "model": model,
                                         "forecast_date": fdate, "location": loc,
                                         "horizon": h, "reference_date": ref,
                                         "quantile": float(q), "value": val,
                                         "y_true": float(y), "phase_origin": "Rising"})
    df = pd.DataFrame(rows)
    path = str(tmp_path / "long.parquet")
    df.to_parquet(path, index=False)
    return path


def test_missing_raf_degrades_gracefully(tmp_path):
    # A dump with base models but no 'raf' (legacy) must not raise.
    bias = {m: 5.0 for m in ENSEMBLE_COMPONENTS}
    v = compute(_long(bias, tmp_path), n_boot=50)
    assert v["earned"] is False
    assert "note" in v and "raf" in v["note"]
    assert set(v["conditions"]) == {"c1_beats_field", "c2_significant",
                                    "c3_backfill", "c4_calibration"}


def test_raf_leader_sets_c1(tmp_path):
    # RAF centered on truth (bias 0), everyone else biased away => RAF is WIS leader.
    bias = {m: 15.0 for m in ENSEMBLE_COMPONENTS}
    bias.update({"raf": 0.0, "raf_noback": 12.0})
    v = compute(_long(bias, tmp_path), n_boot=100)
    assert v["model"] == "raf"
    assert v["conditions"]["c1_beats_field"] is True          # outright lowest WIS
    assert v["ablation_raf_noback_wis"] is not None
    assert set(v["conditions"]) == {"c1_beats_field", "c2_significant",
                                    "c3_backfill", "c4_calibration"}
    assert isinstance(v["earned"], bool)

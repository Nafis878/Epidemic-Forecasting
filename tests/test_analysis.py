"""Tests for STEP 6 analysis utilities: phase labelling and the MIST ablation."""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.visualizer import PHASES, attach_phase, label_phases  # noqa: E402
from models.mist_transformer import MISTNet  # noqa: E402


def _bell(loc="US", n=40, scale=1000.0):
    dates = pd.date_range("2023-09-30", periods=n, freq="7D")
    t = np.linspace(0, np.pi, n)
    return pd.DataFrame({"location": loc, "reference_date": dates,
                         "value": scale * np.sin(t) ** 2 + 5})


def test_label_phases_covers_all_phases_and_peak_at_max():
    truth = _bell()
    lab = label_phases(truth)
    assert set(lab["phase"]).issubset(set(PHASES))
    # All three phases should appear on a full rise-peak-fall curve.
    assert set(lab["phase"]) == set(PHASES)
    # The labelled peak region should sit around the curve's maximum.
    peak_dates = lab.loc[lab["phase"] == "Peak", "reference_date"]
    argmax_date = truth.loc[truth["value"].idxmax(), "reference_date"]
    assert (peak_dates.min() <= argmax_date <= peak_dates.max())


def test_attach_phase_joins_labels():
    truth = _bell()
    results = pd.DataFrame({
        "model": "x", "location": "US",
        "reference_date": truth["reference_date"].iloc[[5, 20, 35]].values,
        "wis": [1.0, 2.0, 3.0],
    })
    merged = attach_phase(results, truth)
    assert "phase" in merged.columns
    assert merged["phase"].notna().all()


def test_local_seasonal_peak_not_global():
    """A small early wave should still get a Peak label even if a later wave is taller."""
    dates = pd.date_range("2023-01-07", periods=60, freq="7D")
    v = np.full(60, 50.0)
    v[8:16] = [100, 200, 350, 400, 350, 200, 120, 80]      # small early wave
    v[40:48] = [400, 1200, 2500, 3000, 2500, 1200, 500, 200]  # big later wave
    truth = pd.DataFrame({"location": "US", "reference_date": dates, "value": v})
    lab = label_phases(truth)
    early = lab.iloc[8:16]
    assert (early["phase"] == "Peak").any(), "local early-wave peak was missed"


def test_mist_ablation_forward_runs():
    """use_mechanistic=False must still produce valid (B,S,H,Q) output."""
    B, S, C, H, Q = 2, 4, 16, 4, 23
    net = MISTNet(C, H, Q, use_mechanistic=False)
    out = net(torch.randn(B, S, C), torch.rand(B, S, C) * 100, torch.eye(S))
    assert out.shape == (B, S, H, Q)

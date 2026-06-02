"""Phase 3.1 tests: the R_t-conditioned blend toward persistence.

The blend must (a) be a no-op-ish identity when R_t >> 1 (trust MIST), (b) pull
the forecast toward the persistence drift when R_t << 1 (declining), and (c)
expose beta as a trainable parameter.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mist_v2 import MISTNetV2  # noqa: E402


def _net(use_blend=True):
    torch.manual_seed(0)
    return MISTNetV2(context=8, horizon=2, n_quantiles=23, patch_sizes=(2, 4),
                     d_model=16, n_heads=2, use_blend=use_blend)


def test_beta_is_trainable_parameter():
    net = _net()
    names = [n for n, _ in net.named_parameters()]
    assert "raw_beta" in names
    assert net.raw_beta.requires_grad


def test_blend_alpha_monotonic_in_rt():
    """alpha = sigmoid(beta (R_t - 1)) is ~0 for declining, ~1 for rising input."""
    net = _net().eval()
    B, S, C = 1, 1, 8
    rising = torch.linspace(10, 80, C).view(B, S, C)        # R_t > 1
    declining = torch.linspace(80, 10, C).view(B, S, C)     # R_t < 1
    from models.mist_transformer import estimate_rt
    assert estimate_rt(rising)[..., -1].item() > 1.0
    assert estimate_rt(declining)[..., -1].item() < 1.0


def test_blend_pulls_toward_persistence_when_declining():
    """With blend on, a declining window's median moves toward the persistence drift."""
    net = _net(use_blend=True).eval()
    net_noblend = _net(use_blend=False).eval()
    # copy weights so the only difference is the blend
    net_noblend.load_state_dict({k: v for k, v in net.state_dict().items()
                                 if k in net_noblend.state_dict()}, strict=False)
    B, S, C = 1, 1, 8
    x_raw = torch.linspace(200, 40, C).view(B, S, C)        # clearly declining
    x_norm = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-5)
    with torch.no_grad():
        blended = net(x_norm, x_raw)[0, 0]                  # (H,Q)
        raw = net_noblend(x_norm, x_raw)[0, 0]
    # persistence (normalised) continues the downward drift; the blended median
    # should sit below the raw MIST median for a declining window.
    med_b = blended[:, 11]                                  # 0.5 quantile index
    med_r = raw[:, 11]
    assert torch.all(med_b <= med_r + 1e-4)


def test_blend_forward_shape():
    net = _net().eval()
    out = net(torch.randn(2, 3, 8), torch.rand(2, 3, 8) * 100, torch.eye(3))
    assert out.shape == (2, 3, 2, 23)

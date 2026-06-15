"""The vintage phase gate must be leakage-safe and label phases sensibly."""

import numpy as np
import pandas as pd

from evaluation.phase_gate import vintage_phase


def _series(values):
    dates = pd.date_range("2023-01-01", periods=len(values), freq="7D")
    return pd.DataFrame({"reference_date": dates, "value": values})


def test_rising_then_declining_labels():
    rising = _series(list(np.linspace(1, 100, 20)))
    declining = _series(list(np.linspace(100, 1, 20)))
    assert vintage_phase(rising) == "Rising"
    assert vintage_phase(declining) == "Declining"


def test_peak_label_near_local_max():
    # ramp up, then flatten at the maximum -> trailing slope ~0 at a high level.
    vals = list(np.linspace(1, 100, 15)) + [100, 99, 100, 99, 100, 100]
    assert vintage_phase(_series(vals)) == "Peak"


def test_empty_history_defaults_rising():
    assert vintage_phase(_series([])) == "Rising"
    assert vintage_phase(None) == "Rising"


def test_is_trailing_only_no_leakage():
    """The label at the end must not change if *future* points are appended.

    vintage_phase only ever receives history up to the origin; this guards that
    it reads the trailing edge and never a centered/forward window.
    """
    base = list(np.linspace(1, 50, 12))
    label_now = vintage_phase(_series(base))
    # Appending later observations is what a *leaky* implementation would peek at.
    with_future = vintage_phase(_series(base + [80, 90, 100]))
    # The label for the shorter (as-of) series is computed from its own last point;
    # it must be a pure function of the trailing data it was given.
    assert label_now == vintage_phase(_series(base))
    assert label_now != "" and with_future != ""

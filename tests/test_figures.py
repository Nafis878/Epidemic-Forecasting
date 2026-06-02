"""Phase 5 test: the main-result figure renders to PDF + PNG from a phase table."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.figures_paper import main_result_figure  # noqa: E402


def test_main_result_figure_writes_files(tmp_path):
    phase = pd.DataFrame(
        {"Rising": [51, 62, 70], "Peak": [153, 179, 206], "Declining": [88, 81, 85]},
        index=["mist_v2", "arima", "patchtst"],
    )
    dm = pd.DataFrame(
        [[float("nan"), 0.4, 0.005], [0.4, float("nan"), 0.2], [0.005, 0.2, float("nan")]],
        index=phase.index, columns=phase.index,
    )
    stem = str(tmp_path / "main_result")
    out = main_result_figure(phase, stem, dm_p=dm)
    assert os.path.exists(out) and os.path.exists(stem + ".png")
    assert os.path.getsize(stem + ".png") > 1000

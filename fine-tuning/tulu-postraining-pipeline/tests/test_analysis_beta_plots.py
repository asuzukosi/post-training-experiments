"""unit tests for beta/kl and displacement helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.analysis import (
    BetaArmPoint,
    DisplacementSeries,
    detect_displacement,
    plot_beta_vs_kl_winrate,
    plot_displacement,
    plot_displacement_arms,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
needs_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed",
)


def test_displacement_series_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        DisplacementSeries(
            beta=0.05,
            steps=[0, 1],
            chosen_logps=[-1.0],
            rejected_logps=[-2.0, -2.1],
        )


def test_detect_displacement() -> None:
    series = DisplacementSeries(
        beta=0.05,
        steps=[0, 10, 20],
        chosen_logps=[-1.0, -1.2, -1.5],
        rejected_logps=[-2.0, -2.3, -2.6],
    )
    flag = detect_displacement(series)
    assert flag["displacement"] is True
    assert flag["chosen_fell"] is True
    assert flag["rejected_fell"] is True


def test_plot_beta_requires_arms() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_beta_vs_kl_winrate([])


@needs_matplotlib
def test_plot_beta_vs_kl_winrate(tmp_path: Path) -> None:
    out = plot_beta_vs_kl_winrate(
        [
            BetaArmPoint(beta=0.05, kl=0.12, win_rate_vs_sft=0.58, win_rate_vs_sft_lc=0.52),
            BetaArmPoint(beta=0.1, kl=0.07, win_rate_vs_sft=0.55, win_rate_vs_sft_lc=0.51),
        ],
        tmp_path / "beta_kl.png",
    )
    assert out.is_file()
    assert out.stat().st_size > 0


@needs_matplotlib
def test_plot_displacement(tmp_path: Path) -> None:
    series = DisplacementSeries(
        beta=0.05,
        steps=[0, 10, 20],
        chosen_logps=[-1.0, -1.2, -1.5],
        rejected_logps=[-2.0, -2.3, -2.6],
    )
    out = plot_displacement(series, tmp_path / "disp.png")
    assert out.is_file()


@needs_matplotlib
def test_plot_displacement_arms(tmp_path: Path) -> None:
    series_list = [
        DisplacementSeries(
            beta=0.05,
            steps=[0, 1, 2],
            chosen_logps=[-1.0, -1.1, -1.4],
            rejected_logps=[-2.0, -2.2, -2.5],
        ),
        DisplacementSeries(
            beta=0.1,
            steps=[0, 1, 2],
            chosen_logps=[-1.0, -1.05, -1.1],
            rejected_logps=[-2.0, -2.05, -2.1],
        ),
    ]
    out = plot_displacement_arms(series_list, tmp_path / "disp_arms.png")
    assert out.is_file()

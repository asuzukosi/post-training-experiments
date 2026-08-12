"""unit tests for chattiness plot helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.analysis import (
    ChattinessPoint,
    chattiness_point_from_style_report,
    plot_length_markdown,
    plot_raw_vs_length_controlled,
    summarize_chattiness,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
needs_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed",
)


def _style_report(
    *,
    mean_chars: float,
    markdown_rate: float,
    win_raw: float,
    win_lc: float,
    delta: float,
) -> dict:
    return {
        "style_b": {"mean_chars": mean_chars, "markdown_rate": markdown_rate},
        "raw": {"win_rate_b": win_raw},
        "length_controlled": {"win_rate_b": win_lc},
        "mean_char_delta_b_minus_a": delta,
    }


def test_point_from_style_report_and_drop() -> None:
    point = chattiness_point_from_style_report(
        "dpo-b0.05",
        _style_report(
            mean_chars=1200,
            markdown_rate=0.6,
            win_raw=0.58,
            win_lc=0.52,
            delta=180,
        ),
    )
    assert point.stage == "dpo-b0.05"
    assert point.win_rate_raw == pytest.approx(0.58)
    assert point.win_rate_lc == pytest.approx(0.52)
    assert point.win_rate_drop == pytest.approx(0.06)
    assert point.mean_char_delta_vs_ref == pytest.approx(180)

    rows = summarize_chattiness([point])
    assert rows[0]["win_rate_drop"] == pytest.approx(0.06)


def test_plot_requires_points() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_raw_vs_length_controlled([])
    with pytest.raises(ValueError, match="non-empty"):
        plot_length_markdown([])


@needs_matplotlib
def test_plot_raw_vs_length_controlled(tmp_path: Path) -> None:
    points = [
        ChattinessPoint(stage="dpo-b0.05", win_rate_raw=0.58, win_rate_lc=0.52),
        ChattinessPoint(stage="dpo-b0.1", win_rate_raw=0.55, win_rate_lc=0.51),
        ChattinessPoint(stage="ppo", win_rate_raw=0.57, win_rate_lc=0.53),
    ]
    out = plot_raw_vs_length_controlled(points, tmp_path / "raw_lc.png")
    assert out.is_file()
    assert out.stat().st_size > 0


@needs_matplotlib
def test_plot_length_markdown(tmp_path: Path) -> None:
    points = [
        ChattinessPoint(stage="dpo-b0.05", mean_chars=1200, markdown_rate=0.6),
        ChattinessPoint(stage="dpo-b0.1", mean_chars=1100, markdown_rate=0.5),
        ChattinessPoint(stage="ppo", mean_chars=1150, markdown_rate=0.55),
    ]
    out = plot_length_markdown(points, tmp_path / "len_md.png")
    assert out.is_file()

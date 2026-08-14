"""unit tests for kl-frontier peak/bound (no gpu)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from analysis.kl_frontier import (
    FrontierPoint,
    build_kl_frontier,
    detect_gold_peak,
    plot_gold_vs_kl,
    plot_inverted_u,
    points_from_sweep,
    write_kl_frontier,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
needs_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed",
)


def _pt(kl: float, gold: float, gold_lc: float | None = None, n: int | None = None) -> FrontierPoint:
    return FrontierPoint(kl=kl, gold=gold, gold_lc=gold_lc, n=n, proxy=kl)


def test_detect_peak_is_first_max() -> None:
    points = [
        _pt(0.0, 0.50, n=1),
        _pt(math.log(4), 0.70, n=4),
        _pt(math.log(8), 0.70, n=8),
        _pt(math.log(16), 0.55, n=16),
    ]
    peak = detect_gold_peak(points, metric="gold")
    assert peak is not None
    assert peak.n == 4
    assert peak.gold == pytest.approx(0.70)
    assert peak.shape == "decline"


def test_plateau_when_later_points_hold() -> None:
    points = [
        _pt(0.0, 0.50, n=1),
        _pt(1.0, 0.62, n=4),
        _pt(2.0, 0.62, n=8),
    ]
    peak = detect_gold_peak(points, metric="gold")
    assert peak is not None
    assert peak.n == 4
    assert peak.shape == "plateau"


def test_peak_at_max_kl_when_still_rising() -> None:
    points = [_pt(0.0, 0.50, n=1), _pt(0.7, 0.55, n=2), _pt(1.4, 0.60, n=4)]
    peak = detect_gold_peak(points, metric="gold")
    assert peak is not None
    assert peak.n == 4
    assert peak.shape == "peak_at_max_kl"


def test_bound_prefers_lc() -> None:
    points = [
        _pt(0.0, 0.50, 0.50, n=1),
        _pt(0.7, 0.80, 0.58, n=2),
        _pt(1.4, 0.70, 0.52, n=4),
    ]
    frontier = build_kl_frontier(points)
    assert frontier.bound is not None
    assert frontier.bound.metric == "gold_lc"
    assert frontier.bound.n == 2
    assert frontier.peak_raw is not None
    assert frontier.peak_raw.n == 2


def test_points_from_sweep_and_write(tmp_path: Path) -> None:
    sweep = {
        "n_values": [1, 2, 4],
        "points": [
            {
                "n": 1,
                "kl": 0.0,
                "gold_win_rate": 0.5,
                "gold_win_rate_lc": 0.5,
                "mean_proxy": 0.1,
            },
            {
                "n": 2,
                "kl": 0.69,
                "gold_win_rate": 0.66,
                "gold_win_rate_lc": 0.60,
                "mean_proxy": 0.3,
            },
            {
                "n": 4,
                "kl": 1.39,
                "gold_win_rate": 0.58,
                "gold_win_rate_lc": 0.54,
                "mean_proxy": 0.5,
            },
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep), encoding="utf-8")
    frontier = build_kl_frontier(points_from_sweep(path))
    assert frontier.bound is not None
    assert frontier.bound.shape == "decline"
    out = write_kl_frontier(frontier, tmp_path / "kl_frontier.json")
    payload = json.loads(out.read_text())
    assert payload["bound"]["n"] == 2



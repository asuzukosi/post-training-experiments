"""unit tests for length/markdown metrics and length-controlled win-rates."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.io import append_jsonl
from eval.judge import FIRST_MODEL, MODEL_TIE, SECOND_MODEL
from eval.style import (
    compute_win_rates,
    has_markdown,
    is_length_matched,
    markdown_hit_count,
    report_head_to_head_style,
    report_head_to_head_style_from_jsonl,
    summarize_style,
)


def _record(
    item_id: str,
    completion_a: str,
    completion_b: str,
    winner: str,
    *,
    pref_ab: float | None = None,
    pref_ba: float | None = None,
) -> dict:
    """minimal judgearena wrapper row."""
    if pref_ab is None:
        pref_ab = {"A": 0.0, "B": 1.0, "tie": 0.5}[winner]
    if pref_ba is None:
        pref_ba = pref_ab
    return {
        "id": item_id,
        "completion_a": completion_a,
        "completion_b": completion_b,
        "winner": winner,
        "pref_ab": pref_ab,
        "pref_ba": pref_ba,
        "judge_backend": "judgearena",
    }




def test_is_length_matched() -> None:
    assert is_length_matched("aaaa", "aaab", max_rel_diff=0.1) is True
    assert is_length_matched("a" * 100, "a" * 50, max_rel_diff=0.1) is False


def test_compute_win_rates() -> None:
    rates = compute_win_rates(
        [
            _record("0", "a0", "b0", FIRST_MODEL),
            _record("1", "a1", "b1", SECOND_MODEL),
            _record("2", "a2", "b2", SECOND_MODEL),
            _record("3", "a3", "b3", MODEL_TIE),
        ]
    )
    assert rates.wins_a == 1
    assert rates.wins_b == 2
    assert rates.ties == 1
    assert rates.win_rate_b == pytest.approx(2 / 3)
    assert rates.win_rate_b_with_ties == pytest.approx((2 + 0.5) / 4)


def test_compute_win_rates_rejects_homemade_row() -> None:
    with pytest.raises(ValueError, match="winner"):
        compute_win_rates([{"completion_a": "a", "completion_b": "b"}])
    with pytest.raises(ValueError, match="completion_a"):
        compute_win_rates([{"winner": FIRST_MODEL, "completion_b": "b"}])
    with pytest.raises(ValueError, match="winner"):
        compute_win_rates(
            [{"winner": "left", "completion_a": "a", "completion_b": "b"}]
        )


def test_report_head_to_head_style_raw_vs_length_controlled() -> None:
    # pair0: B wins, B much longer -> excluded from LC
    # pair1: B wins, similar length -> kept in LC
    # pair2: A wins, similar length -> kept in LC
    records = [
        _record("0", "short", "x" * 200, SECOND_MODEL),
        _record("1", "aaaaaa", "bbbbbb", SECOND_MODEL),
        _record("2", "cccccc", "dddddd", FIRST_MODEL),
    ]
    report = report_head_to_head_style(records, max_rel_length_diff=0.1)
    assert report.n_total == 3
    assert report.n_length_matched == 2
    assert report.raw.wins_b == 2
    assert report.raw.wins_a == 1
    assert report.raw.win_rate_b == pytest.approx(2 / 3)
    assert report.length_controlled.wins_b == 1
    assert report.length_controlled.wins_a == 1
    assert report.length_controlled.win_rate_b == pytest.approx(0.5)
    assert report.mean_char_delta_b_minus_a > 0
    assert report.style_b.mean_chars > report.style_a.mean_chars



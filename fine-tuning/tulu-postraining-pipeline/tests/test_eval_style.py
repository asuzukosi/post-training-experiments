"""unit tests for length/markdown metrics and length-controlled win-rates."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval.io import append_jsonl
from pipeline.eval.style import (
    compute_win_rates,
    has_markdown,
    is_length_matched,
    markdown_hit_count,
    report_head_to_head_style,
    report_head_to_head_style_from_jsonl,
    summarize_style,
)


def test_markdown_detection() -> None:
    plain = "just a sentence with no formatting"
    md = "## Title\n\n- item one\n- item two\n\n**bold** and `code`"
    assert has_markdown(plain) is False
    assert has_markdown(md) is True
    assert markdown_hit_count(md) >= 3


def test_summarize_style() -> None:
    summary = summarize_style(
        [
            "short",
            "## longer markdown response with words",
        ]
    )
    assert summary.n == 2
    assert summary.mean_chars > 0
    assert summary.markdown_rate == 0.5


def test_is_length_matched() -> None:
    assert is_length_matched("aaaa", "aaab", max_rel_diff=0.1) is True
    assert is_length_matched("a" * 100, "a" * 50, max_rel_diff=0.1) is False


def test_compute_win_rates() -> None:
    rates = compute_win_rates(
        [
            {"winner": "A"},
            {"winner": "B"},
            {"winner": "B"},
            {"winner": "tie"},
        ]
    )
    assert rates.wins_a == 1
    assert rates.wins_b == 2
    assert rates.ties == 1
    assert rates.win_rate_b == pytest.approx(2 / 3)
    assert rates.win_rate_b_with_ties == pytest.approx((2 + 0.5) / 4)


def test_report_head_to_head_style_raw_vs_length_controlled() -> None:
    # pair0: B wins, B much longer -> excluded from LC
    # pair1: B wins, similar length -> kept in LC
    # pair2: A wins, similar length -> kept in LC
    records = [
        {
            "id": "0",
            "completion_a": "short",
            "completion_b": "x" * 200,
            "winner": "B",
        },
        {
            "id": "1",
            "completion_a": "aaaaaa",
            "completion_b": "bbbbbb",
            "winner": "B",
        },
        {
            "id": "2",
            "completion_a": "cccccc",
            "completion_b": "dddddd",
            "winner": "A",
        },
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


def test_report_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "head-to-head.jsonl"
    append_jsonl(
        path,
        {
            "id": "1",
            "completion_a": "hello world",
            "completion_b": "hello there",
            "winner": "A",
        },
    )
    report = report_head_to_head_style_from_jsonl(path)
    assert report.n_total == 1
    assert report.raw.wins_a == 1
    d = report.to_dict()
    assert "raw" in d and "length_controlled" in d

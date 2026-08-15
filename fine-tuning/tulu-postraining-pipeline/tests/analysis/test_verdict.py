"""judged comparisons: did one arm beat another, and is the margin real?

the assertions are about DIRECTION and about refusing to call a winner on an
overlapping interval — a flipped sign or a spurious winner is the failure that survives
review, because every other symptom is loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import (
    WinRate,
    assess_head_to_head,
    compare_arms,
    load_win_rate,
    write_verdict,
)


def test_rate_and_interval_come_from_decisive_pairs() -> None:
    """ties carry no signal, so they leave the denominator."""
    w = WinRate(arm="dpo", opponent="sft", wins=300, losses=200, ties=50)
    assert w.decisive == 500
    assert w.rate == pytest.approx(0.6)
    lo, hi = w.ci95
    assert hi - lo == pytest.approx(2 * 1.96 * w.stderr)


def test_interval_narrows_as_prompts_increase() -> None:
    """this is why the prompt count is the lever, not repeated runs."""
    small = WinRate(arm="a", opponent="sft", wins=55, losses=45)
    large = WinRate(arm="a", opponent="sft", wins=550, losses=450)
    assert large.stderr < small.stderr


def test_all_ties_is_an_error_not_a_50_percent_score() -> None:
    with pytest.raises(ValueError, match="every pair was a tie"):
        _ = WinRate(arm="a", opponent="sft", wins=0, losses=0, ties=40).rate


def test_head_to_head_names_the_winner_not_a_position() -> None:
    """the winner is an arm name; 'a'/'b' would mean nothing to a reader."""
    v = assess_head_to_head(WinRate(arm="rs_sft", opponent="dpo", wins=300, losses=200))
    assert v.winner == "rs_sft"
    assert v.decided is True
    assert v.margin == pytest.approx(0.1)


def test_head_to_head_names_the_opponent_when_the_arm_loses() -> None:
    v = assess_head_to_head(WinRate(arm="rs_sft", opponent="dpo", wins=200, losses=300))
    assert v.winner == "dpo"


def test_head_to_head_needs_the_whole_interval_past_parity() -> None:
    """a rate above 0.5 is not a win if the interval still touches it."""
    v = assess_head_to_head(WinRate(arm="rs_sft", opponent="dpo", wins=26, losses=24))
    assert v.winner is None
    assert v.decided is False


def test_compare_arms_margin_is_challenger_minus_baseline() -> None:
    ppo = WinRate(arm="ppo", opponent="sft", wins=320, losses=180)
    dpo = WinRate(arm="dpo", opponent="sft", wins=250, losses=250)
    v = compare_arms(ppo, dpo)
    assert v.margin == pytest.approx(0.14)
    assert v.winner == "ppo"


def test_compare_arms_calls_a_tie_when_intervals_overlap() -> None:
    ppo = WinRate(arm="ppo", opponent="sft", wins=256, losses=244)
    dpo = WinRate(arm="dpo", opponent="sft", wins=252, losses=248)
    v = compare_arms(ppo, dpo)
    assert v.winner is None
    assert v.ci95_low < 0 < v.ci95_high


def test_compare_arms_refuses_a_different_opponent() -> None:
    """dpo-vs-sft against ppo-vs-base measures two different things."""
    ppo = WinRate(arm="ppo", opponent="base", wins=320, losses=180)
    dpo = WinRate(arm="dpo", opponent="sft", wins=250, losses=250)
    with pytest.raises(ValueError, match="different opponents"):
        compare_arms(ppo, dpo)


def test_load_reads_the_arm_and_opponent_off_the_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "model_a": "/ckpt/sft",
                "model_b": "/ckpt/ppo",
                "reports": [
                    {"raw": {"wins_a": 40, "wins_b": 55, "ties": 5}},
                    {"raw": {"wins_a": 45, "wins_b": 50, "ties": 5}},
                ],
            }
        ),
        encoding="utf-8",
    )
    w = load_win_rate(summary)
    assert (w.arm, w.opponent) == ("ppo", "sft")
    assert (w.wins, w.losses, w.ties) == (105, 85, 10)


def test_write_emits_json_and_markdown(tmp_path: Path) -> None:
    v = compare_arms(
        WinRate(arm="ppo", opponent="sft", wins=320, losses=180),
        WinRate(arm="dpo", opponent="sft", wins=250, losses=250),
    )
    out = write_verdict(v, tmp_path / "v.json", markdown_path=tmp_path / "v.md")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["winner"] == "ppo"
    assert payload["decided"] is True
    assert payload["challenger"]["n_decisive"] == 500
    assert "ppo" in (tmp_path / "v.md").read_text(encoding="utf-8")

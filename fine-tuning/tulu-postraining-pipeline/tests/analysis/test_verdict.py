"""judged win-rate verdicts.

the interval comes from the number of prompts judged. these assert direction and the
refusal to call a winner on an overlapping interval — a flipped sign or a spurious
winner is the failure that survives review, because every other symptom is loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import (
    WinRate,
    compare,
    compare_to_chance,
    load_win_rate,
    write_verdict,
)


def test_rate_and_interval_come_from_decisive_pairs() -> None:
    """ties are no signal, so they leave the denominator."""
    w = WinRate(name="dpo", wins=300, losses=200, ties=50)
    assert w.decisive == 500
    assert w.rate == pytest.approx(0.6)
    lo, hi = w.ci95
    assert lo < 0.6 < hi
    assert hi - lo == pytest.approx(2 * 1.96 * w.stderr)


def test_interval_narrows_as_prompts_increase() -> None:
    """this is why the prompt count is the lever, not repeated runs."""
    small = WinRate(name="a", wins=55, losses=45)
    large = WinRate(name="a", wins=550, losses=450)
    assert large.stderr < small.stderr


def test_all_ties_is_an_error_not_a_50_percent_score() -> None:
    with pytest.raises(ValueError, match="no decisive pairs"):
        _ = WinRate(name="a", wins=0, losses=0, ties=40).rate


def test_delta_is_b_minus_a_and_names_the_leader() -> None:
    a = WinRate(name="dpo", wins=250, losses=250)
    b = WinRate(name="ppo", wins=320, losses=180)
    v = compare(a, b, question="dpo vs ppo")
    assert v.delta == pytest.approx(0.14)
    assert v.winner == "b"
    assert "ppo ahead" in v.note


def test_overlapping_intervals_are_a_tie() -> None:
    a = WinRate(name="dpo", wins=252, losses=248)
    b = WinRate(name="ppo", wins=256, losses=244)
    v = compare(a, b, question="dpo vs ppo")
    assert v.winner == "tie"
    assert v.ci95_low < 0 < v.ci95_high


def test_against_chance_needs_the_whole_interval_above_half() -> None:
    """a mean above 0.5 is not a win if the interval still touches it."""
    clear = compare_to_chance(WinRate(name="rs", wins=300, losses=200), question="rs")
    assert clear.winner == "b"

    marginal = compare_to_chance(WinRate(name="rs", wins=26, losses=24), question="rs")
    assert marginal.winner == "tie"


def test_load_sums_every_report_in_a_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "reports": [
                    {"raw": {"wins_a": 40, "wins_b": 55, "ties": 5}},
                    {"raw": {"wins_a": 45, "wins_b": 50, "ties": 5}},
                ]
            }
        ),
        encoding="utf-8",
    )
    w = load_win_rate("dpo", summary)
    assert (w.wins, w.losses, w.ties) == (105, 85, 10)


def test_write_emits_json_and_markdown(tmp_path: Path) -> None:
    v = compare(
        WinRate(name="dpo", wins=250, losses=250),
        WinRate(name="ppo", wins=320, losses=180),
        question="dpo vs ppo",
    )
    out = write_verdict(v, tmp_path / "v.json", markdown_path=tmp_path / "v.md")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["winner"] == "b"
    assert payload["a"]["n_decisive"] == 500
    assert "dpo vs ppo" in (tmp_path / "v.md").read_text(encoding="utf-8")

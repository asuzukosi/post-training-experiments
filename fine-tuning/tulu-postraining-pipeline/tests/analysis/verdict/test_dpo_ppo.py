"""unit tests for dpo-vs-ppo equal-data verdict."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import (
    ArmVsSft,
    arm_from_head_to_head_summary,
    build_dpo_ppo_verdict,
    write_dpo_ppo_verdict,
)


def test_verdict_tie_when_cis_overlap() -> None:
    dpo = ArmVsSft(
        name="dpo-b0.1",
        win_rates_raw=[0.54, 0.56, 0.55],
        win_rates_lc=[0.51, 0.52, 0.50],
    )
    ppo = ArmVsSft(
        name="ppo",
        win_rates_raw=[0.55, 0.57, 0.56],
        win_rates_lc=[0.52, 0.53, 0.51],
    )
    verdict = build_dpo_ppo_verdict(dpo, ppo)
    assert verdict.primary_metric == "length_controlled_win_rate_vs_sft"
    assert verdict.primary_winner == "tie"
    assert verdict.length_controlled is not None
    assert verdict.length_controlled.winner == "tie"


def test_verdict_ppo_when_clearly_ahead() -> None:
    dpo = ArmVsSft(
        name="dpo-b0.1",
        win_rates_raw=[0.50, 0.50, 0.50],
        win_rates_lc=[0.48, 0.48, 0.48],
    )
    ppo = ArmVsSft(
        name="ppo",
        win_rates_raw=[0.70, 0.72, 0.71],
        win_rates_lc=[0.68, 0.69, 0.70],
    )
    verdict = build_dpo_ppo_verdict(dpo, ppo)
    assert verdict.primary_winner == "ppo"
    assert verdict.raw.winner == "ppo"


def test_arm_from_summary_and_write(tmp_path: Path, h2h_summary) -> None:
    dpo = arm_from_head_to_head_summary(
        "dpo-b0.1",
        h2h_summary([0.58, 0.57, 0.59], [0.52, 0.51, 0.53]),
        kl=0.08,
    )
    ppo = arm_from_head_to_head_summary(
        "ppo",
        h2h_summary([0.60, 0.61, 0.59], [0.54, 0.55, 0.53]),
        wall_clock_hours=8.0,
    )
    assert dpo.win_rates_raw == pytest.approx([0.58, 0.57, 0.59])
    assert ppo.kl is None

    verdict = build_dpo_ppo_verdict(dpo, ppo)
    out = write_dpo_ppo_verdict(
        verdict,
        tmp_path / "verdict.json",
        markdown_path=tmp_path / "verdict.md",
    )
    payload = json.loads(out.read_text())
    assert payload["dpo_name"] == "dpo-b0.1"
    assert "primary_winner" in payload
    assert (tmp_path / "verdict.md").is_file()
    assert "primary winner" in (tmp_path / "verdict.md").read_text().lower()


def test_arm_requires_raw_rates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ArmVsSft(name="dpo", win_rates_raw=[])

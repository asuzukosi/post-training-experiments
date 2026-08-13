"""unit tests for rs-sft vs dpo verdict."""
from __future__ import annotations

import json
from pathlib import Path

from analysis import (
    ArmVsSft,
    RS_DPO_CLAIM,
    arm_from_head_to_head_summary,
    build_rs_dpo_verdict,
    write_rs_dpo_verdict,
)


def test_rs_dpo_tie_when_ci_includes_half() -> None:
    rs = ArmVsSft(
        name="rs_sft",
        win_rates_raw=[0.52, 0.50, 0.51],
        win_rates_lc=[0.51, 0.49, 0.50],
    )
    verdict = build_rs_dpo_verdict(rs, dpo_name="dpo-b0.1")
    assert verdict.claim == RS_DPO_CLAIM
    assert verdict.primary_metric == "length_controlled_win_rate_rs_vs_dpo"
    assert verdict.primary_winner == "tie"
    assert verdict.judge_bias is None


def test_rs_dpo_rs_when_ci_above_half() -> None:
    rs = ArmVsSft(
        name="rs_sft",
        win_rates_raw=[0.70, 0.72, 0.71],
        win_rates_lc=[0.68, 0.69, 0.70],
    )
    bias = {
        "n": 10,
        "position": {"disagreement_rate": 0.1},
        "length": {"slope": 0.02},
        "self_preference": {"self_pref_rate": 0.6, "n_mixed": 10},
        "logprob": {"agreement_rate": 0.5},
    }
    verdict = build_rs_dpo_verdict(rs, dpo_name="dpo-b0.1", judge_bias=bias)
    assert verdict.primary_winner == "rs_sft"
    assert verdict.raw.winner == "rs_sft"
    assert verdict.judge_bias is not None
    assert verdict.judge_bias["position"]["disagreement_rate"] == 0.1


def test_rs_dpo_write_and_markdown(tmp_path: Path, h2h_summary) -> None:
    rs = arm_from_head_to_head_summary(
        "rs_sft",
        h2h_summary([0.40, 0.39, 0.41], [0.38, 0.37, 0.39]),
    )
    verdict = build_rs_dpo_verdict(rs, dpo_name="dpo-b0.1")
    assert verdict.primary_winner == "dpo"
    out = write_rs_dpo_verdict(
        verdict,
        tmp_path / "rs_verdict.json",
        markdown_path=tmp_path / "rs_verdict.md",
    )
    payload = json.loads(out.read_text())
    assert payload["dpo_name"] == "dpo-b0.1"
    assert "teacher" in payload["claim"]
    md = (tmp_path / "rs_verdict.md").read_text().lower()
    assert "judge bias" in md
    assert "not provided" in md

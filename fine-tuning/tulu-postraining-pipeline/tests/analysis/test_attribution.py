"""unit tests for stage-attribution table builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import (
    build_attribution_delta,
    AttributionRow,
    compute_stage_deltas,
    StageEvaluation,
    build_stage_attribution_table,
    write_stage_attribution_table,
)


def _skills(ifeval: float, mmlu: float) -> dict:
    return {"ifeval_prompt_strict": ifeval, "mmlu_acc": mmlu}


def _judged(mean_chars: float, md: float, win_raw: float) -> dict:
    return {
        "style_b": {"mean_chars": mean_chars, "markdown_rate": md},
        "raw": {"win_rate_b": win_raw},
    }


def test_build_complete_attribution_table(tmp_path: Path) -> None:
    skills = {
        "base": _skills(0.20, 0.40),
        "sft": _skills(0.45, 0.38),
        "dpo-b0.05": _skills(0.44, 0.37),
        "dpo-b0.1": _skills(0.43, 0.37),
        "ppo": _skills(0.44, 0.36),
    }
    judged = {
        "dpo-b0.05": _judged(1200, 0.6, 0.58),
        "dpo-b0.1": _judged(1100, 0.5, 0.55),
        "ppo": _judged(1150, 0.55, 0.57),
    }
    table = build_stage_attribution_table(
        [
            StageEvaluation.from_files(st, benchmarks=skills.get(st), sft_comparison=judged.get(st))
            for st in skills
        ],
        require_complete=True,
    )
    assert table.complete is True
    by_stage = {r.stage: r for r in table.rows}
    assert list(by_stage) == ["base", "sft", "dpo-b0.05", "dpo-b0.1", "ppo"]
    assert by_stage["sft"].ifeval == pytest.approx(0.45)
    assert by_stage["dpo-b0.05"].sft_win_rate == pytest.approx(0.58)
    by_label = {d.label: d for d in table.deltas}
    assert by_label["sft_vs_base"].ifeval == pytest.approx(0.25)
    assert by_label["dpo-b0.05_vs_sft"].mmlu == pytest.approx(-0.01)

    out = write_stage_attribution_table(table, tmp_path / "attr.json")
    payload = json.loads(out.read_text())
    assert payload["complete"] is True
    assert len(payload["rows"]) == 5


def test_incomplete_raises() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        build_stage_attribution_table(
            [StageEvaluation.from_files("base", benchmarks=_skills(0.2, 0.4))],
            require_complete=True,
        )


def test_incomplete_allowed_when_not_required() -> None:
    table = build_stage_attribution_table(
        [
            StageEvaluation.from_files("base", benchmarks=_skills(0.2, 0.4)),
            StageEvaluation.from_files("sft", benchmarks=_skills(0.3, 0.39)),
        ],
        require_complete=False,
    )
    assert table.complete is False
    assert any(m.startswith("dpo-b0.05") for m in table.missing)


def test_stage_deltas_are_current_minus_previous() -> None:
    """sft_vs_base must be sft - base, not base - sft.

    the whole attribution table reads as "what did this stage ADD", so a flipped
    subtraction reports every improvement as a regression and vice versa.
    """
    rows = [
        AttributionRow(stage="base", ifeval=0.30, mmlu=0.50),
        AttributionRow(stage="sft", ifeval=0.45, mmlu=0.48),
    ]
    d = {x.label: x for x in compute_stage_deltas(rows)}["sft_vs_base"]
    assert d.ifeval == pytest.approx(0.15)   # sft improved format
    assert d.mmlu == pytest.approx(-0.02)    # and cost a little mmlu


def test_preference_stage_deltas_are_measured_against_sft_not_base() -> None:
    """dpo/ppo deltas are vs SFT — measuring them vs base would double-count SFT's gain."""
    rows = [
        AttributionRow(stage="base", ifeval=0.30, mmlu=0.50),
        AttributionRow(stage="sft", ifeval=0.45, mmlu=0.48),
        AttributionRow(
            stage="ppo",
            ifeval=0.55,
            mmlu=0.44,
            mean_chars=1200.0,
            markdown_rate=0.8,
        ),
    ]
    d = {x.label: x for x in compute_stage_deltas(rows)}["ppo_vs_sft"]
    assert d.ifeval == pytest.approx(0.10)   # 0.55 - 0.45, NOT 0.55 - 0.30
    assert d.mmlu == pytest.approx(-0.04)    # 0.44 - 0.48


def test_stage_deltas_tolerate_missing_stages_and_missing_metrics() -> None:
    """a stage that has not run yet must be skipped, not crash or report 0.0.

    reporting 0.0 for a missing metric would read as "this stage changed nothing",
    which is a different and wrong claim.
    """
    assert compute_stage_deltas([]) == []
    # base present, sft absent -> nothing is comparable
    assert compute_stage_deltas([AttributionRow(stage="base", ifeval=0.3)]) == []
    # sft present but the metric is None on one side -> None, not 0.0
    rows = [
        AttributionRow(stage="base", ifeval=None, mmlu=0.50),
        AttributionRow(stage="sft", ifeval=0.45, mmlu=0.48),
    ]
    d = {x.label: x for x in compute_stage_deltas(rows)}["sft_vs_base"]
    assert d.ifeval is None
    assert d.mmlu == pytest.approx(-0.02)


def test_build_delta_subtracts_left_from_right() -> None:
    """the first argument is the starting point; the delta is second minus first."""
    before = AttributionRow(stage="base", ifeval=0.20, mmlu=0.60)
    after = AttributionRow(stage="sft", ifeval=0.45, mmlu=0.58)

    d = build_attribution_delta(before, after)
    assert d.stage == "sft" and d.versus == "base"
    assert d.label == "sft_vs_base"
    assert d.ifeval == pytest.approx(0.25)
    assert d.mmlu == pytest.approx(-0.02)


def test_build_delta_is_directional() -> None:
    """swapping the arguments flips every sign — the order carries the meaning."""
    a = AttributionRow(stage="base", ifeval=0.20)
    b = AttributionRow(stage="sft", ifeval=0.45)
    assert build_attribution_delta(a, b).ifeval == pytest.approx(0.25)
    assert build_attribution_delta(b, a).ifeval == pytest.approx(-0.25)


def test_build_delta_carries_the_win_rate_rather_than_subtracting_it() -> None:
    """a win-rate is already a comparison, so there is nothing to subtract from."""
    sft = AttributionRow(stage="sft", ifeval=0.45)
    ppo = AttributionRow(stage="ppo", ifeval=0.50, sft_win_rate=0.61)

    d = build_attribution_delta(sft, ppo)
    assert d.judged_win_rate == pytest.approx(0.61)
    assert d.ifeval == pytest.approx(0.05)
    # sft has no judged report of its own, so a base->sft delta has no win-rate
    assert build_attribution_delta(AttributionRow(stage="base"), sft).judged_win_rate is None

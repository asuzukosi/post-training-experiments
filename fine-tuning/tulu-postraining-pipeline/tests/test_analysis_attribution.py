"""unit tests for stage-attribution table builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.analysis import (
    build_stage_attribution_table,
    write_stage_attribution_table,
)


def _skills(ifeval: float, mmlu: float) -> dict:
    return {"ifeval_prompt_strict": ifeval, "mmlu_acc": mmlu}


def _style(mean_chars: float, md: float, win_raw: float, win_lc: float) -> dict:
    return {
        "style_b": {"mean_chars": mean_chars, "markdown_rate": md},
        "raw": {"win_rate_b": win_raw},
        "length_controlled": {"win_rate_b": win_lc},
    }


def test_build_complete_attribution_table(tmp_path: Path) -> None:
    skills = {
        "base": _skills(0.20, 0.40),
        "sft": _skills(0.45, 0.38),
        "dpo-b0.05": _skills(0.44, 0.37),
        "dpo-b0.1": _skills(0.43, 0.37),
        "ppo": _skills(0.44, 0.36),
    }
    style = {
        "dpo-b0.05": _style(1200, 0.6, 0.58, 0.52),
        "dpo-b0.1": _style(1100, 0.5, 0.55, 0.51),
        "ppo": _style(1150, 0.55, 0.57, 0.53),
    }
    table = build_stage_attribution_table(
        skills=skills,
        style_vs_sft=style,
        require_complete=True,
    )
    assert table.complete is True
    by_stage = {r.stage: r for r in table.rows}
    assert list(by_stage) == ["base", "sft", "dpo-b0.05", "dpo-b0.1", "ppo"]
    assert by_stage["sft"].format_ifeval == pytest.approx(0.45)
    assert by_stage["dpo-b0.05"].style_win_rate_vs_sft_raw == pytest.approx(0.58)
    assert table.deltas["sft_vs_base"]["format_ifeval"] == pytest.approx(0.25)
    assert table.deltas["dpo-b0.05_vs_sft"]["skills_mmlu"] == pytest.approx(-0.01)

    out = write_stage_attribution_table(table, tmp_path / "attr.json")
    payload = json.loads(out.read_text())
    assert payload["complete"] is True
    assert len(payload["rows"]) == 5


def test_incomplete_raises() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        build_stage_attribution_table(
            skills={"base": _skills(0.2, 0.4)},
            require_complete=True,
        )


def test_incomplete_allowed_when_not_required() -> None:
    table = build_stage_attribution_table(
        skills={"base": _skills(0.2, 0.4), "sft": _skills(0.3, 0.39)},
        require_complete=False,
    )
    assert table.complete is False
    assert any(m.startswith("dpo-b0.05") for m in table.missing)

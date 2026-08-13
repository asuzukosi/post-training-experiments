"""unit tests for judgearena wrapper (no gpu / no real judgearena)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.judge import (
    aggregate_winner,
    judge_incremental,
    judgearena_model_id,
    judgment_id,
    preference_to_winner,
)

def test_judgearena_model_id() -> None:
    assert (
        judgearena_model_id("Qwen/Qwen2.5-32B-Instruct")
        == "VLLM/Qwen/Qwen2.5-32B-Instruct"
    )
    assert (
        judgearena_model_id("VLLM/Qwen/Qwen2.5-14B-Instruct")
        == "VLLM/Qwen/Qwen2.5-14B-Instruct"
    )


def test_preference_to_winner() -> None:
    assert preference_to_winner(0.0) == "A"
    assert preference_to_winner(1.0) == "B"
    assert preference_to_winner(0.5) == "tie"
    assert preference_to_winner(None) == "tie"
    assert preference_to_winner(float("nan")) == "tie"


def test_aggregate_winner() -> None:
    assert aggregate_winner("A", "A") == "A"
    assert aggregate_winner("B", "B") == "B"
    assert aggregate_winner("tie", "tie") == "tie"
    assert aggregate_winner("A", "B") == "tie"
    assert aggregate_winner("A", "tie") == "tie"


def test_judgment_id() -> None:
    assert judgment_id("p0", run=2) == "p0__r2"


def _pair(pid: str) -> dict:
    return {
        "id": judgment_id(pid, run=1),
        "prompt": f"q{pid}",
        "completion_a": f"ans a {pid}",
        "completion_b": f"ans b {pid}",
        "model_a": "sft",
        "model_b": "dpo",
        "run": 1,
    }


def test_judge_incremental_fake_arena_and_resume(
    tmp_path: Path, install_judgearena_stub
) -> None:
    path = tmp_path / "judge.jsonl"
    items = [_pair("p0"), _pair("p1")]
    prefs = {
        "qp0": (0.0, 0.0),
        "qp1": (1.0, 1.0),
        "qp2": (0.0, 1.0),
    }
    judged = install_judgearena_stub(prefs)

    written = judge_incremental(
        items,
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        output_path=path,
        batch_size=2,
    )
    assert len(written) == 2
    assert written[0]["winner"] == "A"
    assert written[0]["order_ab"] == "A"
    assert written[0]["order_ba"] == "A"
    assert written[0]["judge_backend"] == "judgearena"
    assert written[0]["judge_backend_id"] == "VLLM/Qwen/Qwen2.5-32B-Instruct"
    assert written[1]["winner"] == "B"
    assert written[1]["raw_ab"] == "1.0"

    more = items + [_pair("p2")]
    written2 = judge_incremental(
        more,
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        output_path=path,
    )
    assert len(written2) == 1
    assert written2[0]["winner"] == "tie"
    assert judged == ["qp0", "qp1", "qp2"]

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["id"] == "p0__r1"


def test_judge_incremental_requires_completions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completion_a"):
        judge_incremental(
            [{"id": "x", "prompt": "p", "completion_b": "b"}],
            output_path=tmp_path / "j.jsonl",
        )

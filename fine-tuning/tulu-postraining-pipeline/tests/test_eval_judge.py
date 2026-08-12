"""unit tests for pairwise judge helpers (no gpu / no vllm)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval.judge import (
    aggregate_winner,
    judge_incremental,
    judgment_id,
    parse_pairwise_verdict,
    verdict_to_winner,
)


def test_parse_pairwise_verdict() -> None:
    assert parse_pairwise_verdict("1") == "1"
    assert parse_pairwise_verdict("2") == "2"
    assert parse_pairwise_verdict("tie") == "tie"
    assert parse_pairwise_verdict("I prefer 2 overall.") == "2"
    assert parse_pairwise_verdict("") == "tie"
    assert parse_pairwise_verdict("unclear") == "tie"


def test_verdict_to_winner_position_maps() -> None:
    assert verdict_to_winner("1", first_is_a=True) == "A"
    assert verdict_to_winner("2", first_is_a=True) == "B"
    assert verdict_to_winner("1", first_is_a=False) == "B"
    assert verdict_to_winner("2", first_is_a=False) == "A"
    assert verdict_to_winner("tie", first_is_a=True) == "tie"


def test_aggregate_winner() -> None:
    assert aggregate_winner("A", "A") == "A"
    assert aggregate_winner("B", "B") == "B"
    assert aggregate_winner("tie", "tie") == "tie"
    assert aggregate_winner("A", "B") == "tie"
    assert aggregate_winner("A", "tie") == "tie"


def test_judgment_id() -> None:
    assert judgment_id("p0", run=2) == "p0__r2"


def test_judge_incremental_position_swap_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    items = [
        {
            "id": judgment_id("p0", run=1),
            "prompt": "q0",
            "completion_a": "ans a0",
            "completion_b": "ans b0",
            "model_a": "sft",
            "model_b": "dpo",
            "run": 1,
        },
        {
            "id": judgment_id("p1", run=1),
            "prompt": "q1",
            "completion_a": "ans a1",
            "completion_b": "ans b1",
            "model_a": "sft",
            "model_b": "dpo",
            "run": 1,
        },
    ]
    # scripted outputs: for each pair, ab then ba
    # p0: ab says 1 (->A), ba says 2 (->A) => winner A
    # p1: ab says 2 (->B), ba says 1 (->B) => winner B
    scripted = ["1", "2", "2", "1"]
    idx = {"i": 0}

    def fake_gen(prompts: list[str]) -> list[str]:
        out = []
        for _ in prompts:
            out.append(scripted[idx["i"]])
            idx["i"] += 1
        return out

    written = judge_incremental(
        items,
        judge_model="fake-judge",
        output_path=path,
        batch_size=2,
        generate_fn=fake_gen,
    )
    assert len(written) == 2
    assert written[0]["winner"] == "A"
    assert written[0]["order_ab"] == "A"
    assert written[0]["order_ba"] == "A"
    assert written[1]["winner"] == "B"
    assert idx["i"] == 4

    # resume: only new id is judged
    more = items + [
        {
            "id": judgment_id("p2", run=1),
            "prompt": "q2",
            "completion_a": "a2",
            "completion_b": "b2",
            "model_a": "sft",
            "model_b": "dpo",
            "run": 1,
        }
    ]
    # disagreement => tie
    scripted.extend(["1", "1"])  # ab->A, ba first_is_a=False so 1->B => tie
    written2 = judge_incremental(
        more,
        judge_model="fake-judge",
        output_path=path,
        generate_fn=fake_gen,
    )
    assert len(written2) == 1
    assert written2[0]["winner"] == "tie"

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["id"] == "p0__r1"


def test_judge_incremental_requires_completions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completion_a"):
        judge_incremental(
            [{"id": "x", "prompt": "p", "completion_b": "b"}],
            output_path=tmp_path / "j.jsonl",
            generate_fn=lambda ps: ["1"] * len(ps),
        )

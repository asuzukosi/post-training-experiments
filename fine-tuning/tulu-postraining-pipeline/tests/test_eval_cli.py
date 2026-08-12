"""unit tests for eval cli wiring / head-to-head join (no gpu)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval.cli import _selected_modes, _skills_tasks, parse_args
from pipeline.eval.head_to_head import build_judge_items, model_ref, run_head_to_head
from pipeline.eval.io import append_jsonl


def test_parse_args_modes() -> None:
    args = parse_args(["--ifeval", "--mmlu", "--model", "m"])
    assert _selected_modes(args) == ["skills"]
    assert _skills_tasks(args) == ["ifeval", "mmlu"]

    args = parse_args(["--mmlu", "--model", "m"])
    assert _skills_tasks(args) == ["mmlu"]

    args = parse_args(["--reward-bench", "--rm-checkpoint", "rm"])
    assert _selected_modes(args) == ["reward_bench"]

    args = parse_args(
        ["--head-to-head", "--a", "a", "--b", "b", "--prompts", "p.jsonl", "--runs", "2"]
    )
    assert _selected_modes(args) == ["head_to_head"]
    assert args.runs == 2


def test_main_requires_mode() -> None:
    from pipeline.eval.cli import main

    assert main([]) == 2


def test_model_ref_hub_vs_local(tmp_path: Path) -> None:
    assert model_ref("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B"
    local = tmp_path / "ckpt"
    local.mkdir()
    # absolute path
    assert model_ref(local) == str(local)


def test_build_judge_items() -> None:
    prompts = [{"id": "1", "prompt": "hi"}]
    gens_a = [{"id": "1", "completion": "a"}]
    gens_b = [{"id": "1", "completion": "b"}]
    pairs = build_judge_items(
        prompts,
        gens_a,
        gens_b,
        model_a="ma",
        model_b="mb",
        run=2,
    )
    assert pairs[0]["id"] == "1__r2"
    assert pairs[0]["completion_a"] == "a"
    assert pairs[0]["completion_b"] == "b"


def test_run_head_to_head_mocked(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    append_jsonl(prompts, {"id": "p0", "prompt": "hello"})
    out = tmp_path / "h2h"

    def fake_gen(batch):
        return [f"out:{p[:10]}" for p in batch]

    # ab then ba for the single pair
    scripted = ["1", "2"]
    i = {"n": 0}

    def fake_judge(batch):
        out_texts = []
        for _ in batch:
            out_texts.append(scripted[i["n"] % len(scripted)])
            i["n"] += 1
        return out_texts

    summary = run_head_to_head(
        model_a="fake-a",
        model_b="fake-b",
        prompts_path=prompts,
        output_dir=out,
        runs=1,
        judge_model="fake-judge",
        apply_chat_template=False,
        generate_fn=fake_gen,
        judge_generate_fn=fake_judge,
    )
    assert summary["runs"] == 1
    assert (out / "summary_fake-a_vs_fake-b.json").is_file()
    style = json.loads((out / "style_fake-a_vs_fake-b_r1.json").read_text())
    assert style["raw"]["wins_a"] == 1

"""unit tests for head-to-head join (no gpu)."""
from __future__ import annotations

import json
from pathlib import Path

from eval.head_to_head import build_judge_items, model_ref, run_head_to_head
from eval.io import append_jsonl


def test_model_ref_hub_vs_local(tmp_path: Path) -> None:
    assert model_ref("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B"
    local = tmp_path / "ckpt"
    local.mkdir()
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


def test_run_head_to_head_mocked(
    tmp_path: Path,
    install_judgearena_stub,
    install_vllm_stub,
    install_chat_template_stub,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    append_jsonl(prompts, {"id": "p0", "prompt": "hello"})
    out = tmp_path / "h2h"
    install_judgearena_stub({"hello": (0.0, 0.0)})
    install_vllm_stub()
    install_chat_template_stub()

    summary = run_head_to_head(
        model_a="fake-a",
        model_b="fake-b",
        prompts_path=prompts,
        output_dir=out,
        runs=1,
        judge_model="Qwen/Qwen2.5-32B-Instruct",
    )
    assert summary["runs"] == 1
    assert (out / "summary_fake-a_vs_fake-b.json").is_file()
    style = json.loads((out / "style_fake-a_vs_fake-b_r1.json").read_text())
    assert style["raw"]["wins_a"] == 1

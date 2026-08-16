"""unit tests for head-to-head join (no gpu)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.head_to_head import build_judge_items, run_head_to_head
from prepare.paths import model_ref
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
    )
    assert pairs[0]["id"] == "1"
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
        judge_model="Qwen/Qwen2.5-32B-Instruct",
    )
    assert summary["raw"]["wins_a"] == 1
    summary_path = out / "summary_fake-a_vs_fake-b.json"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text())["raw"]["wins_a"] == 1

    # generations land in the SHARED dir, not the per-comparison one, so the next
    # comparison against fake-a finds them already done
    shared = out.parent / "generations"
    assert (shared / "gens_fake-a.jsonl").is_file()
    assert (shared / "gens_fake-b.jsonl").is_file()
    assert not list(out.glob("gens_*.jsonl"))


def test_generations_are_reused_across_comparisons(
    tmp_path: Path,
    install_judgearena_stub,
    install_vllm_stub,
    install_chat_template_stub,
) -> None:
    """sft appears in four head-to-heads; it must generate once, not four times.

    the shared generations dir is the whole saving — without it each comparison writes
    into its own directory and `generate_incremental` has nothing to skip.
    """
    prompts = tmp_path / "prompts.jsonl"
    append_jsonl(prompts, {"id": "p0", "prompt": "hello"})
    install_judgearena_stub({"hello": (0.0, 0.0)})
    calls = install_vllm_stub()
    install_chat_template_stub()

    root = tmp_path / "h2h"
    common = dict(
        model_a="sft",
        prompts_path=prompts,
        judge_model="Qwen/Qwen2.5-32B-Instruct",
    )
    run_head_to_head(model_b="dpo", output_dir=root / "sft_vs_dpo", **common)
    after_first = len(calls)
    run_head_to_head(model_b="ppo", output_dir=root / "sft_vs_ppo", **common)

    assert after_first == 2, "first comparison generates both sides"
    # only ppo is new; sft is served from the shared dir
    assert len(calls) - after_first == 1
    assert (root / "generations" / "gens_sft.jsonl").is_file()


def test_a_basename_collision_is_caught_not_served(
    tmp_path: Path, install_vllm_stub, install_chat_template_stub
) -> None:
    """the cache's one hazard: two checkpoints whose dirs share a name."""
    from eval.head_to_head import cached_generations

    install_vllm_stub()
    install_chat_template_stub()
    gens_dir = tmp_path / "generations"
    items = [{"id": "p0", "prompt": "hello"}]

    cached_generations(items, model="runs/a/ckpt", gens_dir=gens_dir)
    with pytest.raises(ValueError, match="share a basename"):
        cached_generations(items, model="runs/b/ckpt", gens_dir=gens_dir)

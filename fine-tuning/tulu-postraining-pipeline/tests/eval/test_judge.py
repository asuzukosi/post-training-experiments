"""unit tests for judgearena wrapper (no gpu / no real judgearena)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eval.judge import (
    aggregate_winner,
    judge_incremental,
    judgearena_model_id,
    judgment_id,
    preference_to_winner,
)


def test_aggregate_winner() -> None:
    assert aggregate_winner("A", "A") == "A"
    assert aggregate_winner("B", "B") == "B"
    assert aggregate_winner("tie", "tie") == "tie"
    assert aggregate_winner("A", "B") == "tie"
    assert aggregate_winner("A", "tie") == "tie"


def _pair(prompt_id: str) -> dict:
    return {
        "id": judgment_id(prompt_id, run=1),
        "prompt": f"q{prompt_id}",
        "completion_a": f"ans a {prompt_id}",
        "completion_b": f"ans b {prompt_id}",
        "model_a": "sft",
        "model_b": "dpo",
        "run": 1,
    }


def test_judge_incremental_requires_completions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completion_a"):
        judge_incremental(
            [{"id": "x", "prompt": "p", "completion_b": "b"}],
            output_path=tmp_path / "j.jsonl",
        )


def test_judge_resume_survives_a_killed_process(
    tmp_path: Path, install_judgearena_stub
) -> None:
    """the judge is the longest-running eval job, so it is the most likely to be killed.

    same failure as generate_incremental: a torn last line is only recoverable while
    it is still last. one more append merges two records and the file is unreadable.
    """
    path = tmp_path / "judge.jsonl"
    items = [_pair("p0"), _pair("p1"), _pair("p2")]
    install_judgearena_stub({"qp0": (0.0, 0.0), "qp1": (1.0, 1.0), "qp2": (0.0, 1.0)})

    judge_incremental(
        items[:1],
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        output_path=path,
        batch_size=8,
    )
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "p1__r1", "winner": "A", "pref')

    written = judge_incremental(
        items,
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        output_path=path,
        batch_size=8,
    )
    assert len(written) == 2
    rows = [json.loads(line) for line in path.read_text().strip().splitlines()]
    assert [r["id"] for r in rows] == [judgment_id(p, run=1) for p in ("p0", "p1", "p2")]


def test_judge_temperature_reaches_sampling_params(install_judgearena_stub) -> None:
    """temp 0 must land on sampling_params, not on the engine.

    judgearena 0.1.0 hardcodes SamplingParams(temperature=0.6) and forwards spare
    kwargs to vllm's LLM(), which rejects `temperature`. so passing it through
    make_model both raises AND, but for the raise, would leave the judge sampling at
    0.6 — position-swapped passes then disagree at random and aggregate_winner turns
    real wins into ties.
    """
    from eval.judge import build_judge_llm

    install_judgearena_stub({})
    llm = build_judge_llm("Qwen/Qwen2.5-1.5B-Instruct", temperature=0.0)
    assert llm.sampling_params.temperature == 0.0
    assert llm.sampling_params.top_p == 1.0


def test_judge_refuses_a_backend_without_sampling_params(monkeypatch) -> None:
    """never judge at an unknown temperature — that is the failure we cannot see."""
    import types

    from eval import judge as judge_mod

    utils = types.ModuleType("judgearena.utils")
    utils.make_model = lambda *a, **k: object()  # no sampling_params
    monkeypatch.setitem(sys.modules, "judgearena.utils", utils)
    with pytest.raises(RuntimeError, match="sampling_params"):
        judge_mod.build_judge_llm("Qwen/Qwen2.5-1.5B-Instruct", temperature=0.0)


def test_judge_engine_is_built_once_across_batches(
    tmp_path: Path, install_judgearena_stub, monkeypatch
) -> None:
    """one judge engine for many batches.

    judge_incremental calls score_with_judgearena per batch, and a judge engine reserves
    ~90% of the card just like a generation one — so a per-batch build leaves the second
    batch with nowhere to load. this is the same defect that was in vllm_backend.
    """
    from eval import vllm_backend

    install_judgearena_stub({f"qp{i}": (0.0, 0.0) for i in range(6)})
    builds: list[str] = []
    real_live = vllm_backend.live_engine

    def counting_live(key, factory):
        if vllm_backend._LIVE is None or vllm_backend._LIVE[0] != key:
            builds.append(key)
        return real_live(key, factory)

    monkeypatch.setattr("eval.vllm_backend.live_engine", counting_live)

    items = [_pair(f"p{i}") for i in range(6)]
    judge_incremental(
        items,
        judge_model="Qwen/Qwen2.5-1.5B-Instruct",
        output_path=tmp_path / "judge.jsonl",
        batch_size=2,
    )
    assert builds == ["judge:VLLM/Qwen/Qwen2.5-1.5B-Instruct"]


def test_judge_evicts_a_generation_engine(monkeypatch) -> None:
    """head_to_head generates then judges in ONE process; both cannot hold the card."""
    from eval import vllm_backend

    released: list[str] = []
    vllm_backend._LIVE = ("generate:model-a", object())
    monkeypatch.setattr(vllm_backend, "release_engine",
                        lambda: released.append(vllm_backend._LIVE[0]))
    vllm_backend.live_engine("judge:VLLM/j", lambda: object())
    assert released == ["generate:model-a"]

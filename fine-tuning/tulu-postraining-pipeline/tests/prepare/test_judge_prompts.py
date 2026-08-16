"""the judging set must never share a prompt with what ppo trains on.

that overlap is invisible once it happens — ppo simply scores better on prompts it was
rl-trained on, and the win-rate reads as a real gain.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_tools import build_ultrafeedback_prompt_pool, prompt_ids_of
from eval.head_to_head import load_prompt_items
from prepare.judge import prepare_judge_prompts


def _rows(n: int) -> list[dict]:
    return [
        {
            "prompt": f"question number {i} about something entirely unremarkable",
            "prompt_id": f"p{i}",
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        }
        for i in range(n)
    ]


@pytest.fixture
def ppo_cfg(tmp_path: Path) -> dict:
    return {
        "dataset": "unused",
        "split": "test_prefs",
        "num_prompts": 6,
        "seed": 42,
        "processed_path": str(tmp_path / "ppo_pool"),
    }


def _write_ppo_pool(cfg: dict, rows: list[dict]) -> set[str]:
    from datasets import Dataset

    pool = build_ultrafeedback_prompt_pool(
        rows, num_prompts=int(cfg["num_prompts"]), seed=int(cfg["seed"])
    )
    Dataset.from_list(pool).save_to_disk(cfg["processed_path"])
    return prompt_ids_of(pool)


def test_judge_prompts_exclude_the_ppo_pool(monkeypatch, tmp_path: Path, ppo_cfg) -> None:
    rows = _rows(10)
    ppo_ids = _write_ppo_pool(ppo_cfg, rows)
    monkeypatch.setattr("prepare.judge.load_ultrafeedback", lambda cfg, **kw: rows)

    out = prepare_judge_prompts(
        ppo_cfg, output_path=tmp_path / "eval_prompts.jsonl", skip_decontam=True
    )

    items = load_prompt_items(out)
    ids = {r["id"] for r in items}
    assert len(items) == len(rows) - len(ppo_ids)  # everything ppo leaves, not a target size
    assert not ids & ppo_ids
    assert all(r["prompt"] for r in items)


def test_judge_prompts_are_frozen(monkeypatch, tmp_path: Path, ppo_cfg) -> None:
    """rerunning must reproduce the file byte for byte, or comparisons drift apart."""
    rows = _rows(10)
    _write_ppo_pool(ppo_cfg, rows)
    monkeypatch.setattr("prepare.judge.load_ultrafeedback", lambda cfg, **kw: rows)

    first = (tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    for path in first:
        prepare_judge_prompts(ppo_cfg, output_path=path, skip_decontam=True)
    assert first[0].read_text() == first[1].read_text()


def test_limit_truncates_but_stays_disjoint(monkeypatch, tmp_path: Path, ppo_cfg) -> None:
    rows = _rows(10)
    ppo_ids = _write_ppo_pool(ppo_cfg, rows)
    monkeypatch.setattr("prepare.judge.load_ultrafeedback", lambda cfg, **kw: rows)

    out = prepare_judge_prompts(
        ppo_cfg, output_path=tmp_path / "small.jsonl", num_prompts=2, skip_decontam=True
    )
    items = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(items) == 2
    assert not {r["id"] for r in items} & ppo_ids

"""unit tests for incremental vllm generation helpers (no gpu / no vllm)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval import (
    append_jsonl,
    generate_incremental,
    load_completed_ids,
    pending_items,
)


def test_load_completed_ids_missing_file(tmp_path: Path) -> None:
    assert load_completed_ids(tmp_path / "missing.jsonl") == set()


def test_append_and_load_completed_ids(tmp_path: Path) -> None:
    path = tmp_path / "gens.jsonl"
    append_jsonl(path, {"id": "a", "completion": "one"})
    append_jsonl(path, {"id": "b", "completion": "two"})
    assert load_completed_ids(path) == {"a", "b"}


def test_pending_items_skips_completed() -> None:
    items = [
        {"id": "1", "prompt": "p1"},
        {"id": "2", "prompt": "p2"},
        {"id": "3", "prompt": "p3"},
    ]
    pending = pending_items(items, {"2"})
    assert [x["id"] for x in pending] == ["1", "3"]


def test_generate_incremental_writes_and_skips(tmp_path: Path) -> None:
    path = tmp_path / "out" / "gens.jsonl"
    items = [
        {"id": "p0", "prompt": "hello 0", "source": "t"},
        {"id": "p1", "prompt": "hello 1"},
        {"id": "p2", "prompt": "hello 2"},
    ]
    calls: list[list[str]] = []

    def fake_gen(prompts: list[str]) -> list[str]:
        calls.append(list(prompts))
        return [f"out:{p}" for p in prompts]

    written = generate_incremental(
        items,
        model="fake-model",
        output_path=path,
        batch_size=2,
        generate_fn=fake_gen,
    )
    assert [r["id"] for r in written] == ["p0", "p1", "p2"]
    assert written[0]["completion"] == "out:hello 0"
    assert written[0]["source"] == "t"
    assert written[0]["model"] == "fake-model"
    assert len(calls) == 2
    assert calls[0] == ["hello 0", "hello 1"]
    assert calls[1] == ["hello 2"]

    # resume: only missing id is generated
    calls.clear()
    # drop last line to simulate partial run mid-file... actually full file exists;
    # add a new item and re-run with overlap
    more = items + [{"id": "p3", "prompt": "hello 3"}]
    written2 = generate_incremental(
        more,
        model="fake-model",
        output_path=path,
        batch_size=8,
        generate_fn=fake_gen,
    )
    assert [r["id"] for r in written2] == ["p3"]
    assert calls == [["hello 3"]]

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["p0", "p1", "p2", "p3"]


def test_generate_incremental_noop_when_complete(tmp_path: Path) -> None:
    path = tmp_path / "gens.jsonl"
    items = [{"id": "x", "prompt": "hi"}]
    append_jsonl(path, {"id": "x", "completion": "done", "prompt": "hi", "model": "m"})

    def boom(_prompts: list[str]) -> list[str]:
        raise AssertionError("should not generate when complete")

    written = generate_incremental(
        items,
        model="m",
        output_path=path,
        generate_fn=boom,
    )
    assert written == []


def test_generate_incremental_rejects_bad_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        generate_incremental(
            [{"id": "a", "prompt": "p"}],
            model="m",
            output_path=tmp_path / "x.jsonl",
            batch_size=0,
            generate_fn=lambda ps: [""] * len(ps),
        )


def test_generate_incremental_requires_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt"):
        generate_incremental(
            [{"id": "a"}],
            model="m",
            output_path=tmp_path / "x.jsonl",
            generate_fn=lambda ps: [""] * len(ps),
        )

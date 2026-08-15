"""unit tests for incremental vllm generation helpers (no gpu / no vllm)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from eval import (
    append_jsonl,
    generate_incremental,
    load_completed_ids,
    pending_items,
)


def test_generate_incremental_writes_and_skips(
    tmp_path: Path, install_vllm_stub
) -> None:
    path = tmp_path / "out" / "gens.jsonl"
    items = [
        {"id": "p0", "prompt": "hello 0", "source": "t"},
        {"id": "p1", "prompt": "hello 1"},
        {"id": "p2", "prompt": "hello 2"},
    ]
    calls = install_vllm_stub()
    written = generate_incremental(
        items,
        model="fake-model",
        output_path=path,
        batch_size=2,
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
    more = items + [{"id": "p3", "prompt": "hello 3"}]
    written2 = generate_incremental(
        more,
        model="fake-model",
        output_path=path,
        batch_size=8,
    )
    assert [r["id"] for r in written2] == ["p3"]
    assert calls == [["hello 3"]]

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["p0", "p1", "p2", "p3"]


def test_resume_survives_a_killed_process(
    tmp_path: Path, install_vllm_stub
) -> None:
    """a real pod kill tears the last line; the job must still be resumable.

    truncating on a line boundary (what a hand-rolled test does) never exercises
    this. a torn line is only recoverable while it is still the LAST line — one more
    append merges two records into one and moves the damage into the middle of the
    file, where it is indistinguishable from real corruption.
    """
    path = tmp_path / "gens.jsonl"
    items = [{"id": f"p{i}", "prompt": f"hello {i}"} for i in range(4)]
    calls = install_vllm_stub()
    generate_incremental(items[:2], model="m", output_path=path, batch_size=8)

    # killed mid-append: last record half written, no terminating newline
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "p2", "comple')

    calls.clear()
    written = generate_incremental(items, model="m", output_path=path, batch_size=8)

    assert [r["id"] for r in written] == ["p2", "p3"]
    rows = [json.loads(line) for line in path.read_text().strip().splitlines()]
    assert [r["id"] for r in rows] == ["p0", "p1", "p2", "p3"]


def test_corruption_away_from_the_tail_still_raises(tmp_path: Path) -> None:
    """only the final line is a crash artifact; a bad line mid-file is real damage."""
    from eval import load_completed_ids

    path = tmp_path / "gens.jsonl"
    path.write_text('{"id": "a"}\nnot json at all\n{"id": "c"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid jsonl"):
        load_completed_ids(path)


class _FakeOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [types.SimpleNamespace(text=text)]


@pytest.fixture
def fake_vllm_module(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """install a fake `vllm` and record the model of every LLM built.

    the other stubs replace `vllm_generate` itself, so they cannot see how many
    engines it builds — which is the thing worth guarding here.
    """
    from eval import vllm_backend

    built: list[str] = []

    class FakeLLM:
        def __init__(self, model: str) -> None:
            built.append(model)

        def generate(self, prompts, sampling):
            return [_FakeOutput(f"out:{p}") for p in prompts]

    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(vllm_backend, "_LIVE", None, raising=False)
    return built


def test_engine_is_loaded_once_across_batches(
    tmp_path: Path, fake_vllm_module: list[str]
) -> None:
    """one engine for many write batches.

    `generate_incremental` calls `vllm_generate` once per write batch, so a
    per-call `LLM(...)` reloads weights every `batch_size` prompts — invisible to
    the stubbed tests, and minutes of dead gpu time per hundred prompts.
    """
    items = [{"id": f"p{i}", "prompt": f"hello {i}"} for i in range(6)]
    generate_incremental(
        items,
        model="fake-model",
        output_path=tmp_path / "gens.jsonl",
        batch_size=2,
    )
    assert fake_vllm_module == ["fake-model"]


def test_engine_is_rebuilt_when_model_changes(
    tmp_path: Path, fake_vllm_module: list[str]
) -> None:
    """head-to-head alternates two models; only one engine may be live at a time."""
    items = [{"id": "p0", "prompt": "hello"}]
    for model in ("model-a", "model-b"):
        generate_incremental(
            items,
            model=model,
            output_path=tmp_path / f"gens_{model}.jsonl",
            batch_size=8,
        )
    assert fake_vllm_module == ["model-a", "model-b"]

"""unit tests for eval decontam bank construction (mocked hub loads).

asserts intended eval splits are used and mmlu_auxiliary_train is excluded.
"""
from __future__ import annotations

from typing import Any

import pytest

from prepare.decontam import build_eval_decontam_bank


class _ColDataset:
    """minimal stand-in for a datasets.Dataset column access."""

    def __init__(self, rows: list[dict[str, Any]], field: str) -> None:
        self._rows = rows
        self._field = field

    def __getitem__(self, key: str) -> list[str]:
        if key != self._field:
            raise KeyError(key)
        return [r[self._field] for r in self._rows]


def test_build_eval_decontam_bank_sources_and_excludes_aux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    mmlu_gets: list[str] = []

    rb = _ColDataset(
        [{"prompt": "one two three four five six seven eight"}],
        "prompt",
    )
    ifeval = _ColDataset(
        [{"prompt": "ifeval wants exact format output right now please"}],
        "prompt",
    )
    mmlu_splits = {
        "test": _ColDataset(
            [{"question": "mmlu test asks about ancient world history facts"}],
            "question",
        ),
        "validation": _ColDataset(
            [{"question": "mmlu validation covers cell biology mitosis stages now"}],
            "question",
        ),
        "dev": _ColDataset(
            [{"question": "mmlu dev covers basic algebra word problems today"}],
            "question",
        ),
        # present on the hub object but must never be read into the bank
        "auxiliary_train": _ColDataset(
            [{"question": "aux train stem must never enter default bank"}],
            "question",
        ),
    }

    class _MmluDict(dict):
        def __getitem__(self, key: str) -> Any:
            mmlu_gets.append(key)
            return super().__getitem__(key)

    def fake_load_dataset(*args: Any, **kwargs: Any) -> Any:
        load_calls.append((args, kwargs))
        name = args[0] if args else kwargs.get("path")
        if name == "allenai/reward-bench":
            assert kwargs.get("split") == "filtered"
            return rb
        if name == "google/IFEval":
            assert kwargs.get("split") == "train"
            return ifeval
        if name == "cais/mmlu":
            assert args[1] == "all" or kwargs.get("name") == "all"
            return _MmluDict(mmlu_splits)
        raise AssertionError(f"unexpected dataset load: {args=} {kwargs=}")

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    bank = build_eval_decontam_bank()

    loaded_names = [args[0] for args, _ in load_calls]
    assert loaded_names == ["allenai/reward-bench", "google/IFEval", "cais/mmlu"]
    assert set(mmlu_gets) == {"test", "validation", "dev"}
    assert "auxiliary_train" not in mmlu_gets

    expected = {
        "one two three four five six seven eight",
        "ifeval wants exact format output right now please",
        "mmlu test asks about ancient world history facts",
        "mmlu validation covers cell biology mitosis stages now",
        "mmlu dev covers basic algebra word problems today",
    }
    missing = expected - bank
    assert not missing, f"missing ngrams: {missing}"
    assert "aux train stem must never enter default bank" not in bank
    assert len(bank) == 5

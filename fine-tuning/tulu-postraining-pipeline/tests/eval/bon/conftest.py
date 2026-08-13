"""shared bon test fixtures (no gpu)."""
from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture
def candidate() -> Callable[..., dict]:
    def _make(
        prompt_id: str,
        sample_idx: int,
        completion: str,
        *,
        prompt: str = "write a bio",
    ) -> dict:
        return {
            "id": f"{prompt_id}__{sample_idx}",
            "prompt_id": prompt_id,
            "prompt": prompt,
            "completion": completion,
            "sample_idx": sample_idx,
        }

    return _make


@pytest.fixture
def lexicographic_score():
    def _score(batch, *, judge_model, temperature):
        out = []
        for item in batch:
            pref = 1.0 if item["completion_b"] > item["completion_a"] else 0.0
            out.append(
                {"pref_ab": pref, "pref_ba": pref, "raw_ab": "x", "raw_ba": "x"}
            )
        return out

    return _score

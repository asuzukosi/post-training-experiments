"""unit tests for bon candidate grouping."""
from __future__ import annotations

import pytest

from eval.bon import group_candidates


def test_group_candidates_assigns_missing_sample_idx() -> None:
    rows = [
        {"prompt_id": "p0", "prompt": "q", "completion": "b", "id": "p0-b"},
        {"prompt_id": "p0", "prompt": "q", "completion": "a", "id": "p0-a"},
        {"prompt_id": "p1", "prompt": "r", "completion": "x", "id": "p1-x"},
    ]
    grouped = group_candidates(rows)
    assert list(grouped) == ["p0", "p1"]
    assert [c["completion"] for c in grouped["p0"]] == ["a", "b"]
    assert [c["sample_idx"] for c in grouped["p0"]] == [0, 1]


def test_group_candidates_rejects_mismatched_prompts(candidate) -> None:
    with pytest.raises(ValueError, match="mismatched"):
        group_candidates(
            [
                candidate("p0", 0, "a", prompt="q1"),
                candidate("p0", 1, "b", prompt="q2"),
            ]
        )

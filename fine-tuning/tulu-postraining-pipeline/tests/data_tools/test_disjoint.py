"""unit tests for rm/dpo prompt_id disjointness (no network / no gpu).

spec requires rm and dpo pair sets stay disjoint by prompt_id.
"""
from __future__ import annotations

import pytest

from data_tools.ultrafeedback import assert_disjoint_prompt_ids, prompt_ids_of


def _row(prompt_id: str) -> dict:
    return {"prompt_id": prompt_id, "chosen": [], "rejected": []}


def test_assert_disjoint_prompt_ids_passes_when_disjoint() -> None:
    left = [_row("rm-1"), _row("rm-2")]
    right = [_row("dpo-1"), _row("dpo-2")]
    assert_disjoint_prompt_ids(left, right)
    assert prompt_ids_of(left).isdisjoint(prompt_ids_of(right))


def test_assert_disjoint_prompt_ids_raises_on_overlap() -> None:
    left = [_row("shared"), _row("rm-only")]
    right = [_row("shared"), _row("dpo-only")]
    with pytest.raises(ValueError, match="prompt_id overlap between sets \\(1 ids\\)"):
        assert_disjoint_prompt_ids(left, right)


def test_assert_disjoint_prompt_ids_reports_multiple_overlaps() -> None:
    left = [_row("a"), _row("b"), _row("c")]
    right = [_row("b"), _row("c"), _row("d")]
    with pytest.raises(ValueError, match="prompt_id overlap between sets \\(2 ids\\)"):
        assert_disjoint_prompt_ids(left, right)


def test_assert_disjoint_prompt_ids_empty_sets() -> None:
    assert_disjoint_prompt_ids([], [_row("x")])
    assert_disjoint_prompt_ids([_row("x")], [])
    assert_disjoint_prompt_ids([], [])

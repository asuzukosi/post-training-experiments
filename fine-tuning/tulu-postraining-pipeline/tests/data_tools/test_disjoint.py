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


class _FakeDS:
    """minimal stand-in: to_trl_preference_columns only uses these two members.

    avoids constructing a real Dataset, which cannot be built on python 3.14
    (datasets/dill pickling bug) — so this test runs in every environment.
    """

    def __init__(self, columns):
        self.column_names = list(columns)
        self.removed = None

    def remove_columns(self, cols):
        self.removed = sorted(cols)
        return _FakeDS([c for c in self.column_names if c not in cols])


def test_to_trl_preference_columns_drops_only_the_extras() -> None:
    """trl accepts only {chosen,rejected} or {prompt,chosen,rejected} as message lists.

    our prepared artifact also keeps prompt_id (disjointness assert), score_chosen /
    score_rejected (margin analysis) and a *string* prompt. that matches neither shape,
    so trl raises KeyError inside a .map() and it surfaces much later as an unrelated
    "text input must be of type str" from the tokenizer.
    """
    from data_tools.ultrafeedback import to_trl_preference_columns

    ds = _FakeDS(
        ["prompt", "prompt_id", "messages", "chosen", "rejected", "score_chosen", "score_rejected"]
    )
    out = to_trl_preference_columns(ds)
    assert sorted(out.column_names) == ["chosen", "rejected"]
    assert ds.removed == ["messages", "prompt", "prompt_id", "score_chosen", "score_rejected"]


def test_to_trl_preference_columns_is_a_noop_when_already_reduced() -> None:
    from data_tools.ultrafeedback import to_trl_preference_columns

    ds = _FakeDS(["chosen", "rejected"])
    out = to_trl_preference_columns(ds)
    assert out is ds and ds.removed is None  # no needless .map()/copy

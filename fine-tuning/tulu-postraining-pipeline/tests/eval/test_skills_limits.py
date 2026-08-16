"""per-task eval depth: mmlu at 25/subject, ifeval whole.

one lm-eval `limit` is global across the task list, but the two axes need different
depths — 1,425 mmlu questions resolves the 5-point alarm, while truncating ifeval's 541
would quietly weaken the format axis. these guard the resolution and the grouping.
"""
from __future__ import annotations

import pytest

from eval.lm_eval_skills import (
    DEFAULT_TASK_LIMITS,
    group_tasks_by_limit,
    normalise_limit,
    resolve_task_limits,
)


def test_defaults_give_mmlu_25_and_ifeval_everything() -> None:
    limits = resolve_task_limits(["ifeval", "mmlu"])
    assert limits == {"ifeval": None, "mmlu": 25}


@pytest.mark.parametrize(
    "value,expected",
    [("all", None), ("none", None), ("", None), (None, None), ("25", 25), (25.0, 25), (0.5, 0.5)],
)
def test_normalise_limit(value, expected) -> None:
    assert normalise_limit(value) == expected


@pytest.mark.parametrize("bad", ["twenty", 0, -5])
def test_normalise_limit_rejects_nonsense(bad) -> None:
    with pytest.raises(ValueError):
        normalise_limit(bad)


def test_explicit_task_limits_beat_the_global_limit() -> None:
    """a smoke passes --limit; an explicit per-task depth must still win."""
    limits = resolve_task_limits(
        ["ifeval", "mmlu"], task_limits={"ifeval": "all"}, limit=2
    )
    assert limits == {"ifeval": None, "mmlu": 2}


def test_global_limit_applies_where_no_task_limit_is_given() -> None:
    assert resolve_task_limits(["mmlu"], limit=3) == {"mmlu": 3}


def test_unknown_task_runs_whole() -> None:
    """failing open costs money; failing closed silently weakens the number."""
    assert resolve_task_limits(["gsm8k"]) == {"gsm8k": None}


def test_tasks_sharing_a_limit_share_one_engine_init() -> None:
    groups = group_tasks_by_limit({"ifeval": None, "mmlu": 25, "arc": None})
    assert groups == [(None, ["ifeval", "arc"]), (25, ["mmlu"])]


def test_the_default_map_matches_the_plan() -> None:
    """25 x 57 subjects = 1,425, the count the +/-1.3 point figure was sized from."""
    assert DEFAULT_TASK_LIMITS["mmlu"] == 25
    assert DEFAULT_TASK_LIMITS["ifeval"] is None


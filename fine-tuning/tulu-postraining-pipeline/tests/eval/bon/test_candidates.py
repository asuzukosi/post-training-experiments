"""unit tests for bon candidate grouping."""
from __future__ import annotations

import pytest

from eval.bon import group_candidates



def test_group_candidates_rejects_mismatched_prompts(candidate) -> None:
    with pytest.raises(ValueError, match="mismatched"):
        group_candidates(
            [
                candidate("p0", 0, "a", prompt="q1"),
                candidate("p0", 1, "b", prompt="q2"),
            ]
        )

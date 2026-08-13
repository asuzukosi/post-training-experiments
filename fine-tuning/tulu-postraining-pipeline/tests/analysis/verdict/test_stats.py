"""unit tests for verdict run summaries."""
from __future__ import annotations

import pytest

from analysis import summarize_runs


def test_summarize_runs_ci_widens_for_small_n() -> None:
    s = summarize_runs([0.50, 0.60, 0.55])
    assert s.n == 3
    assert s.mean == pytest.approx(0.55)
    assert s.ci95_low < s.mean < s.ci95_high

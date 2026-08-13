"""shared h2h summary fixture for verdict tests."""
from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture
def h2h_summary() -> Callable[..., dict]:
    def _make(win_raw: list[float], win_lc: list[float]) -> dict:
        reports = []
        for i, (raw, lc) in enumerate(zip(win_raw, win_lc, strict=True), start=1):
            reports.append(
                {
                    "run": i,
                    "raw": {"win_rate_b": raw},
                    "length_controlled": {"win_rate_b": lc},
                }
            )
        return {
            "model_a": "sft",
            "model_b": "arm",
            "runs": len(reports),
            "reports": reports,
        }

    return _make

"""unit tests for flip-rate scoring. no gpu."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.io import append_jsonl
from eval.steer.flip_rate import (
    load_flip_probes,
    contains_answer,
    run_flip_trials,
    score_flip_rate,
    trial_flipped,
)




def test_score_flip_rate_pooled_and_per_probe() -> None:
    records = [
        {
            "id": "p0",
            "answer": "4",
            "trials": [
                {"first": "4", "second": "5"},
                {"first": "4", "second": "4"},
                {"first": "nope", "second": "5"},
            ],
        },
        {
            "id": "p1",
            "answer": "paris",
            "first": "paris",
            "second": "lyon",
        },
    ]
    summary = score_flip_rate(records)
    # eligible: p0 two trials (one flip) + p1 one trial (flip) -> 2/3
    assert summary.n_eligible == 3
    assert summary.n_flipped == 2
    assert summary.flip_rate == pytest.approx(2 / 3)




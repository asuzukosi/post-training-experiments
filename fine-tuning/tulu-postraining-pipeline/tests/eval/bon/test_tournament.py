"""unit tests for bon pairing, ties, and select_top1."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.bon import (
    build_round_pairs,
    group_candidates,
    pair_id,
    pick_winner_candidate,
    select_top1,
    sort_pairs_by_length,
)
from eval.io import append_jsonl
from eval.judge import FIRST_MODEL, MODEL_TIE, SECOND_MODEL





def test_pick_winner_tie_keeps_shorter(candidate) -> None:
    a = candidate("p0", 0, "xxxx")
    b = candidate("p0", 1, "yy")
    winner = pick_winner_candidate({"winner": MODEL_TIE}, a, b)
    assert winner["sample_idx"] == 1
    winner_a = pick_winner_candidate({"winner": FIRST_MODEL}, a, b)
    assert winner_a["sample_idx"] == 0
    winner_b = pick_winner_candidate({"winner": SECOND_MODEL}, a, b)
    assert winner_b["sample_idx"] == 1



def test_select_top1_resumes_completed_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate, lexicographic_score
) -> None:
    path = tmp_path / "pairs.jsonl"
    append_jsonl(
        path,
        {
            "id": pair_id("p0", round_idx=1, sample_a=0, sample_b=1),
            "prompt": "write a bio",
            "completion_a": "aa",
            "completion_b": "bb",
            "winner": SECOND_MODEL,
            "pref_ab": 1.0,
            "pref_ba": 1.0,
        },
    )
    calls = {"n": 0}

    def fake_score(batch, *, judge_model, temperature):
        calls["n"] += len(batch)
        return lexicographic_score(
            batch, judge_model=judge_model, temperature=temperature
        )

    monkeypatch.setattr("eval.judge.score_with_judgearena", fake_score)
    winners = select_top1(
        group_candidates(
            [candidate("p0", 0, "aa"), candidate("p0", 1, "bb")]
        ),
        judge_model="fake-14b",
        judge_path=path,
    )
    assert winners["p0"]["completion"] == "bb"
    assert calls["n"] == 0

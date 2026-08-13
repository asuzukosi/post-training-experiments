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


def test_pair_id_is_stable() -> None:
    assert pair_id("p0", round_idx=2, sample_a=1, sample_b=4) == "p0__t2__1v4"


def test_sort_pairs_by_length_shortest_max_first() -> None:
    pairs = [
        {"id": "long", "completion_a": "aa", "completion_b": "x" * 20},
        {"id": "short", "completion_a": "a", "completion_b": "bb"},
    ]
    ordered = sort_pairs_by_length(pairs)
    assert [p["id"] for p in ordered] == ["short", "long"]


def test_build_round_pairs_bye_and_length_sort(candidate) -> None:
    remaining = {
        "p0": [
            candidate("p0", 0, "short"),
            candidate("p0", 1, "also-short"),
            candidate("p0", 2, "this one is much longer than the others"),
        ]
    }
    pairs, byes = build_round_pairs(remaining, round_idx=1)
    assert byes["p0"]["sample_idx"] == 2
    assert len(pairs) == 1
    assert pairs[0]["sample_idx_a"] == 0
    assert pairs[0]["sample_idx_b"] == 1
    assert pairs[0]["id"] == pair_id("p0", round_idx=1, sample_a=0, sample_b=1)


def test_pick_winner_tie_keeps_shorter(candidate) -> None:
    a = candidate("p0", 0, "xxxx")
    b = candidate("p0", 1, "yy")
    winner = pick_winner_candidate({"winner": MODEL_TIE}, a, b)
    assert winner["sample_idx"] == 1
    winner_a = pick_winner_candidate({"winner": FIRST_MODEL}, a, b)
    assert winner_a["sample_idx"] == 0
    winner_b = pick_winner_candidate({"winner": SECOND_MODEL}, a, b)
    assert winner_b["sample_idx"] == 1


def test_select_top1_lexicographic_and_n1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate, lexicographic_score
) -> None:
    monkeypatch.setattr("eval.judge.score_with_judgearena", lexicographic_score)
    grouped = group_candidates(
        [
            candidate("p0", 0, "aa"),
            candidate("p0", 1, "cc"),
            candidate("p0", 2, "bb"),
            candidate("p0", 3, "dd"),
            candidate("p1", 0, "only"),
        ]
    )
    winners = select_top1(
        grouped,
        judge_model="fake-14b",
        judge_path=tmp_path / "pairs.jsonl",
        batch_size=8,
    )
    assert winners["p0"]["completion"] == "dd"
    assert winners["p1"]["completion"] == "only"
    assert winners["p1"]["sample_idx"] == 0


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

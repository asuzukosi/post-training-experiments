"""best-of-n: grouping, proxy selection, tournament, and the rs-sft row it writes.

best-of-n exists here to build the rejection-sampling arm: generate n candidates per
prompt, pick the best, and train on it. selection must be reproducible and must take
exactly n candidates from a nested ladder, or the arm is not comparable across n.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.bon import (
    build_round_pairs,
    build_rs_sft_row,
    group_candidates,
    pair_id,
    pick_winner_candidate,
    run_bon_selection,
    select_top1,
    sort_pairs_by_length,
)
from eval.bon.proxy import pick_proxy_winner, pool_for_n, select_top1_by_proxy
from eval.io import append_jsonl
from eval.judge import FIRST_MODEL, MODEL_TIE, SECOND_MODEL


def _cand(idx: int, score: float, prompt_id: str = "p0") -> dict:
    return {
        "id": f"{prompt_id}__s{idx}",
        "prompt_id": prompt_id,
        "prompt": "q",
        "completion": f"answer {idx}",
        "sample_idx": idx,
        "proxy_score": score,
    }


def test_pool_for_n_is_exactly_n_and_nested_across_the_ladder() -> None:
    """the n-ladder must be nested prefixes, or selection is not comparable across n."""
    cands = [_cand(i, score=float(i)) for i in range(8)]
    pools = {n: pool_for_n(cands, n, prompt_id="p0") for n in (1, 2, 4, 8)}

    for n, pool in pools.items():
        assert len(pool) == n, f"pool for N={n} had {len(pool)} candidates"
        assert [c["sample_idx"] for c in pool] == list(range(n))

    # nested: every smaller pool is a prefix of every larger one
    for small, large in ((1, 2), (2, 4), (4, 8)):
        assert pools[large][: small] == pools[small]


def test_pool_for_n_refuses_to_silently_shrink() -> None:
    """a short pool must raise, not quietly return fewer than n.

    returning 3 candidates for n=8 would attribute the result to the wrong n, and
    nothing downstream could tell.
    """
    cands = [_cand(i, score=float(i)) for i in range(3)]
    with pytest.raises(ValueError, match=r"missing sample_idx \[3, 4"):
        pool_for_n(cands, 8, prompt_id="p0")
    with pytest.raises(ValueError, match="n must be >= 1"):
        pool_for_n(cands, 0, prompt_id="p0")


def test_pick_proxy_winner_breaks_ties_toward_the_lower_index() -> None:
    """deterministic ties matter: a tie broken at random makes the sweep unreproducible."""
    pool = [_cand(0, 0.7), _cand(1, 0.7), _cand(2, 0.7)]
    assert pick_proxy_winner(pool)["sample_idx"] == 0
    # order of the input must not change the answer
    assert pick_proxy_winner(list(reversed(pool)))["sample_idx"] == 0


def test_build_rs_sft_row(candidate) -> None:
    row = build_rs_sft_row(candidate("p0", 3, "top"), n_candidates=8)
    assert row["messages"][0] == {"role": "user", "content": "write a bio"}
    assert row["messages"][1] == {"role": "assistant", "content": "top"}
    assert row["sample_idx"] == 3
    assert row["n_candidates"] == 8


def test_run_bon_selection_writes_rs_sft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate,
    lexicographic_score,
) -> None:
    gens = tmp_path / "gens.jsonl"
    for row in (
        candidate("p0", 0, "aa"),
        candidate("p0", 1, "zz"),
        candidate("p1", 0, "mm"),
        candidate("p1", 1, "nn"),
    ):
        append_jsonl(gens, row)

    captured: dict[str, list] = {}

    def fake_save(rows, path):
        captured["rows"] = list(rows)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr("eval.judge.score_with_judgearena", lexicographic_score)
    monkeypatch.setattr("eval.bon.select.save_rows", fake_save)

    processed = tmp_path / "rs_sft"
    out = run_bon_selection(
        generations_path=gens,
        output_dir=tmp_path / "bon",
        processed_path=processed,
        judge_model="fake-14b",
    )
    assert out == processed
    rows = captured["rows"]
    assert len(rows) == 2
    by_prompt_id = {r["prompt_id"]: r for r in rows}
    assert by_prompt_id["p0"]["messages"][1]["content"] == "zz"
    assert by_prompt_id["p1"]["messages"][1]["content"] == "nn"
    assert by_prompt_id["p0"]["source"] == "rs_sft"
    assert by_prompt_id["p0"]["n_candidates"] == 2
    assert (tmp_path / "bon" / "bon_selections.jsonl").is_file()


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

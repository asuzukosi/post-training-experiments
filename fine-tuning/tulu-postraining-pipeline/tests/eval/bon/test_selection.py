"""T4 — the BoN selection maths behind the O8 inverted-U.

the KL frontier argument rests on BoN's KL from the reference growing as ~log N. that
only holds if the pool for each N is exactly N candidates AND the ladder is NESTED —
the N=2 pool must be a prefix of the N=4 pool, drawn from the same generations. if the
pools were independently sampled per N, the curve would still plot and would still look
like an inverted-U while measuring something else entirely.
"""
from __future__ import annotations

import pytest

from eval.bon.proxy import pick_proxy_winner, pool_for_n, select_top1_by_proxy
from eval.bon.sweep import build_gold_pairs


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
    """the N-ladder must be nested prefixes, or `KL ~ log N` is not what is measured."""
    cands = [_cand(i, score=float(i)) for i in range(8)]
    pools = {n: pool_for_n(cands, n, prompt_id="p0") for n in (1, 2, 4, 8)}

    for n, pool in pools.items():
        assert len(pool) == n, f"pool for N={n} had {len(pool)} candidates"
        assert [c["sample_idx"] for c in pool] == list(range(n))

    # nested: every smaller pool is a prefix of every larger one
    for small, large in ((1, 2), (2, 4), (4, 8)):
        assert pools[large][: small] == pools[small]


def test_pool_for_n_refuses_to_silently_shrink() -> None:
    """a short pool must raise, not quietly return fewer than N.

    returning 3 candidates for N=8 would put a point on the BoN curve at the wrong N —
    the plot would look fine and the x-axis would be a lie.
    """
    cands = [_cand(i, score=float(i)) for i in range(3)]
    with pytest.raises(ValueError, match=r"missing sample_idx \[3, 4"):
        pool_for_n(cands, 8, prompt_id="p0")
    with pytest.raises(ValueError, match="n must be >= 1"):
        pool_for_n(cands, 0, prompt_id="p0")


def test_pool_for_n_is_indexed_not_positional() -> None:
    """candidates may arrive out of order; selection must key on sample_idx."""
    shuffled = [_cand(2, 0.9), _cand(0, 0.1), _cand(1, 0.5)]
    pool = pool_for_n(shuffled, 2, prompt_id="p0")
    assert [c["sample_idx"] for c in pool] == [0, 1]


def test_pick_proxy_winner_takes_the_highest_score() -> None:
    pool = [_cand(0, 0.1), _cand(1, 0.9), _cand(2, 0.5)]
    assert pick_proxy_winner(pool)["sample_idx"] == 1


def test_pick_proxy_winner_breaks_ties_toward_the_lower_index() -> None:
    """deterministic ties matter: a tie broken at random makes the sweep unreproducible."""
    pool = [_cand(0, 0.7), _cand(1, 0.7), _cand(2, 0.7)]
    assert pick_proxy_winner(pool)["sample_idx"] == 0
    # order of the input must not change the answer
    assert pick_proxy_winner(list(reversed(pool)))["sample_idx"] == 0


def test_pick_proxy_winner_rejects_an_empty_pool_and_missing_scores() -> None:
    with pytest.raises(ValueError, match="empty candidate pool"):
        pick_proxy_winner([])
    unscored = [{"id": "x", "sample_idx": 0, "proxy_score": None}]
    with pytest.raises(ValueError, match="proxy_score"):
        pick_proxy_winner(unscored)


def test_select_top1_is_per_prompt_not_global() -> None:
    """a global argmax would return one winner for the whole sweep."""
    grouped = {
        "p0": [_cand(0, 0.1, "p0"), _cand(1, 0.9, "p0")],
        "p1": [_cand(0, 0.8, "p1"), _cand(1, 0.2, "p1")],
    }
    winners = select_top1_by_proxy(grouped, 2)
    assert set(winners) == {"p0", "p1"}
    assert winners["p0"]["sample_idx"] == 1
    assert winners["p1"]["sample_idx"] == 0


def test_gold_pairs_put_the_n1_baseline_in_slot_a() -> None:
    """a = baseline (N=1), b = selected-at-N. swapping them inverts the gold curve."""
    baseline = {"p0": _cand(0, 0.1), "p1": _cand(0, 0.2, "p1")}
    selected = {"p0": _cand(3, 0.9), "p1": _cand(2, 0.7, "p1")}
    pairs = build_gold_pairs(baseline, selected, n=8)
    assert [p["prompt_id"] for p in pairs] == ["p0", "p1"]  # sorted, stable
    assert pairs[0]["completion_a"] == "answer 0"           # baseline
    assert pairs[0]["completion_b"] == "answer 3"           # selected at N
    assert pairs[0]["id"].endswith("__n8")


def test_gold_pairs_refuse_a_prompt_id_mismatch() -> None:
    """comparing different prompts to each other would silently produce noise."""
    with pytest.raises(ValueError, match="prompt_id mismatch"):
        build_gold_pairs({"p0": _cand(0, 0.1)}, {"p9": _cand(0, 0.2, "p9")}, n=4)

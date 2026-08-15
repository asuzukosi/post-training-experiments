"""T5 — the half of eval/steer that is pure tensor maths (no model needed).

the sign of the contrastive vector is the assertion that matters. v = mean(s+) - mean(s-)
points FROM the negative toward the positive trait, and `apply` adds alpha*v to the
residual stream. flip that sign and steering pushes the model further INTO sycophancy
while the plumbing looks perfect — you would report "steering makes sycophancy worse"
and the number would be real, just measuring the opposite of what O11 claims.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.steer.flip_rate import followup_prompt, require_probe
from eval.steer.vectors import (
    cap_tau,
    contrastive_vector,
    last_token_index,
    load_trait_pairs,
    middle_layer_ids,
    require_trait_pair,
)


def test_contrastive_vector_points_from_negative_to_positive() -> None:
    """v must be mean(pos) - mean(neg), not the reverse."""
    v = contrastive_vector(pos=[[2.0, 0.0]], neg=[[0.0, 0.0]])
    assert v[0] == pytest.approx(1.0)   # +x, the direction of pos
    assert v[1] == pytest.approx(0.0)

    flipped = contrastive_vector(pos=[[0.0, 0.0]], neg=[[2.0, 0.0]])
    assert flipped[0] == pytest.approx(-1.0)


def test_contrastive_vector_is_unit_norm() -> None:
    """alpha is the only magnitude knob; a non-unit v makes alpha uncalibrated."""
    v = contrastive_vector(pos=[[3.0, 4.0]], neg=[[0.0, 0.0]])
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0)
    assert v == pytest.approx([0.6, 0.8])


def test_contrastive_vector_averages_over_examples() -> None:
    v = contrastive_vector(pos=[[2.0, 0.0], [0.0, 2.0]], neg=[[0.0, 0.0], [0.0, 0.0]])
    assert v == pytest.approx([0.70710678, 0.70710678])


def test_contrastive_vector_refuses_degenerate_input() -> None:
    """identical s+/s- means the trait pairs did not separate — a silent zero vector
    would steer by nothing and read as "steering has no effect"."""
    with pytest.raises(ValueError, match="~0"):
        contrastive_vector(pos=[[1.0, 1.0]], neg=[[1.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        contrastive_vector(pos=[], neg=[[1.0]])
    with pytest.raises(ValueError, match="hidden size mismatch"):
        contrastive_vector(pos=[[1.0, 2.0]], neg=[[1.0]])


def test_cap_tau_is_the_25th_percentile_of_projections() -> None:
    """tau caps how far a hidden may be pushed along v; the wrong quantile
    either never clamps (no cap) or clamps everything (no steering)."""
    # projections onto the unit x-axis are just the x components: 0,1,2,3,4
    hiddens = [[float(i), 0.0] for i in range(5)]
    assert cap_tau(hiddens, [1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="need hiddens"):
        cap_tau([], [1.0, 0.0])


def test_middle_layer_ids_is_the_middle_third_and_never_empty() -> None:
    assert middle_layer_ids(24) == list(range(8, 16))
    assert middle_layer_ids(3) == [1]
    # tiny models must still yield at least one layer rather than an empty sweep
    for n in range(1, 8):
        assert len(middle_layer_ids(n)) >= 1
        assert max(middle_layer_ids(n)) < n
    with pytest.raises(ValueError, match="n_layers must be >= 1"):
        middle_layer_ids(0)


def test_last_token_index_assumes_no_left_padding() -> None:
    """documents a real constraint: sum(mask)-1 is the last REAL token only when
    padding is on the right (or absent).

    collect_last_token_hiddens tokenizes one text at a time, so masks are all-ones and
    this is correct today. if that is ever batched with left padding — which is what
    generation uses elsewhere in this repo — this silently selects a PAD position and
    every extracted vector becomes garbage without raising.
    """
    torch = pytest.importorskip("torch")

    unpadded = torch.tensor([[1, 1, 1]])
    assert last_token_index(unpadded).tolist() == [2]

    right_padded = torch.tensor([[1, 1, 1, 0, 0]])
    assert last_token_index(right_padded).tolist() == [2]

    left_padded = torch.tensor([[0, 0, 1, 1, 1]])
    assert last_token_index(left_padded).tolist() == [2]  # WRONG position (real = 4)


def test_trait_pair_validation(tmp_path: Path) -> None:
    ok = require_trait_pair({"id": "t0", "prompt_pos": "a", "prompt_neg": "b"})
    assert ok == {"id": "t0", "prompt_pos": "a", "prompt_neg": "b"}
    for bad in (
        {"prompt_pos": "a", "prompt_neg": "b"},          # no id
        {"id": "t", "prompt_pos": "  ", "prompt_neg": "b"},  # blank
        {"id": "t", "prompt_pos": "a"},                  # no neg
    ):
        with pytest.raises(ValueError):
            require_trait_pair(bad)

    p = tmp_path / "pairs.jsonl"
    p.write_text(
        json.dumps({"id": "t0", "prompt_pos": "a", "prompt_neg": "b"}) + "\n",
        encoding="utf-8",
    )
    assert len(load_trait_pairs(p)) == 1
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no trait pairs"):
        load_trait_pairs(empty)


def test_flip_probe_validation_and_followup_shape() -> None:
    probe = require_probe({"id": "p0", "prompt": "2+2?", "answer": "4"})
    assert probe["answer"] == "4"
    assert probe["pushback"], "a default pushback must be supplied, not left empty"

    with pytest.raises(ValueError, match="answer"):
        require_probe({"id": "p0", "prompt": "q", "answer": "  "})

    # the follow-up must carry the model's OWN first answer, or the pushback turn is
    # not testing whether the model caves under disagreement.
    text = followup_prompt("2+2?", "4", "Are you sure? I think it is 5.")
    assert "2+2?" in text and "assistant: 4" in text and "I think it is 5" in text

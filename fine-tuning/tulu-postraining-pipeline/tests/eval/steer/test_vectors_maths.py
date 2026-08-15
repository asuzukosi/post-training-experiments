"""the half of eval/steer that is pure tensor maths (no model needed).

the sign of the contrastive vector is the assertion that matters. v = mean(s+) - mean(s-)
points FROM the negative toward the positive trait, and `apply` adds alpha*v to the
residual stream. flip that sign and steering pushes the model further INTO sycophancy
while the plumbing looks perfect — you would report "steering makes sycophancy worse"
and the number would be real, just measuring the opposite of the intended effect.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.steer.flip_rate import followup_prompt, require_probe
from eval.steer.extract import last_token_index
from eval.steer.vectors import (
    cap_tau,
    contrastive_vector,
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


def test_contrastive_vector_refuses_degenerate_input() -> None:
    """identical s+/s- means the trait pairs did not separate — a silent zero vector
    would steer by nothing and read as "steering has no effect"."""
    with pytest.raises(ValueError, match="~0"):
        contrastive_vector(pos=[[1.0, 1.0]], neg=[[1.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        contrastive_vector(pos=[], neg=[[1.0]])
    with pytest.raises(ValueError, match="hidden size mismatch"):
        contrastive_vector(pos=[[1.0, 2.0]], neg=[[1.0]])


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

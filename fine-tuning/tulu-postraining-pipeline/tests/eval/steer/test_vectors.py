"""unit tests for contrastive sycophancy vectors. no gpu."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.io import append_jsonl
from eval.steer.vectors import (
    cap_tau,
    contrastive_vector,
    load_trait_pairs,
    load_vectors,
    middle_layer_ids,
    save_vectors,
    SycophancyVectors,
    LayerVector,
)


def test_middle_layer_ids() -> None:
    assert middle_layer_ids(28) == list(range(9, 18))
    with pytest.raises(ValueError, match="n_layers"):
        middle_layer_ids(0)


def test_contrastive_vector_points_at_pos_minus_neg() -> None:
    pos = [[1.0, 0.0], [1.0, 0.0]]
    neg = [[0.0, 0.0], [0.0, 0.0]]
    v = contrastive_vector(pos, neg)
    assert v[0] == pytest.approx(1.0)
    assert v[1] == pytest.approx(0.0)


def test_contrastive_vector_rejects_identical() -> None:
    same = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="~0"):
        contrastive_vector(same, same)


def test_cap_tau_is_25th_percentile() -> None:
    v = [1.0, 0.0]
    hiddens = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    tau = cap_tau(hiddens, v)
    assert 0.0 <= tau <= 1.0


def test_extract_last_token_on_stub_model() -> None:
    import torch

    from eval.steer.vectors import extract_sycophancy_vectors

    class _out:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class _tok:
        def __call__(self, text, return_tensors="pt", add_special_tokens=True):
            n = 2 if "agree" in text else 3
            return {
                "input_ids": torch.ones(1, n, dtype=torch.long),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
            }

    class _model:
        def __init__(self):
            self.config = type("c", (), {"num_hidden_layers": 6})()
            self.device = torch.device("cpu")
            self.training = False

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def __call__(self, attention_mask, output_hidden_states=True, use_cache=False, **_kw):
            seq = int(attention_mask.shape[1])
            last = torch.tensor([1.0, 0.0, 0.0, 0.0] if seq == 2 else [0.0, 0.0, 0.0, 0.0])
            states = []
            for _ in range(7):
                t = torch.zeros(1, seq, 4)
                t[0, -1] = last
                states.append(t)
            return _out(tuple(states))

    vecs = extract_sycophancy_vectors(
        [
            {
                "id": "t0",
                "prompt_pos": "i think 2+2=5, agree?",
                "prompt_neg": "what is 2+2?",
            }
        ],
        model=_model(),
        tokenizer=_tok(),
        model_id="stub",
    )
    assert [lv.layer for lv in vecs.layers] == [2, 3]
    v = vecs.by_layer()[2].vector
    assert v[0] == pytest.approx(1.0)
    assert vecs.by_layer()[2].tau is not None


def test_load_and_save_trait_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    append_jsonl(
        path,
        {
            "id": "t0",
            "prompt_pos": "i think 2+2=5, agree?",
            "prompt_neg": "what is 2+2?",
        },
    )
    pairs = load_trait_pairs(path)
    assert pairs[0]["prompt_pos"].startswith("i think")
    out = tmp_path / "vecs.json"
    save_vectors(
        SycophancyVectors(
            model="fake",
            layers=[
                LayerVector(
                    layer=12, vector=[1.0, 0.0], tau=0.2, n_pos=1, n_neg=1
                )
            ],
        ),
        out,
    )
    loaded = load_vectors(out)
    assert loaded.by_layer()[12].tau == pytest.approx(0.2)

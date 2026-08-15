"""getting hidden states out of a running model - the only forward passes in steer/.

split out of `vectors.py` so the contrastive maths stays cpu-testable and everything
that needs a real model, a tokenizer and gpu memory sits behind one import.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eval.steer.vectors import (
    PROMPT_NEG_KEY,
    PROMPT_POS_KEY,
    LayerVector,
    SycophancyVectors,
    cap_tau,
    contrastive_vector,
    middle_layer_ids,
    require_trait_pair,
)


def last_token_index(attention_mask: Any) -> Any:
    """index of the last non-pad token per row. ASSUMES RIGHT PADDING (or none).

    sum(mask) - 1 is the last real position only when the pads sit at the end.
    `collect_last_token_hiddens` below tokenizes one text at a time so masks are
    all-ones and this holds — but generation elsewhere in this repo uses LEFT padding,
    and under left padding this returns a PAD position and every extracted vector is
    silently garbage. batch this only after fixing the indexing.
    """
    return attention_mask.long().sum(dim=-1) - 1


def collect_last_token_hiddens(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    layers: Sequence[int],
) -> dict[int, list[list[float]]]:
    """forward each text; take residual stream at last prompt token per layer."""
    import torch

    layer_ids = [int(i) for i in layers]
    out: dict[int, list[list[float]]] = {i: [] for i in layer_ids}
    was_training = bool(model.training)
    model.eval()
    try:
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                pred = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = pred.hidden_states
            # hidden_states[0] is embeddings; layer i residual is index i+1
            idx = int(last_token_index(encoded["attention_mask"])[0].item())
            for layer in layer_ids:
                hs_index = layer + 1
                if hs_index >= len(hidden_states):
                    raise ValueError(
                        f"layer {layer} out of range "
                        f"(hidden_states={len(hidden_states)})"
                    )
                vec = hidden_states[hs_index][0, idx].detach().float().cpu().tolist()
                out[layer].append(vec)
    finally:
        if was_training:
            model.train()
    return out


def extract_sycophancy_vectors(
    pairs: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    layers: Sequence[int] | None = None,
    model_id: str = "",
) -> SycophancyVectors:
    """extract v_ℓ on middle layers (or `layers`) from trait pairs."""
    parsed = [require_trait_pair(row, line_no=i) for i, row in enumerate(pairs, start=1)]
    n_layers = int(model.config.num_hidden_layers)
    layer_ids = list(layers) if layers is not None else middle_layer_ids(n_layers)
    pos_texts = [row[PROMPT_POS_KEY] for row in parsed]
    neg_texts = [row[PROMPT_NEG_KEY] for row in parsed]
    pos_h = collect_last_token_hiddens(
        model, tokenizer, pos_texts, layers=layer_ids
    )
    neg_h = collect_last_token_hiddens(
        model, tokenizer, neg_texts, layers=layer_ids
    )
    extracted: list[LayerVector] = []
    for layer in layer_ids:
        vector = contrastive_vector(pos_h[layer], neg_h[layer])
        tau = cap_tau(pos_h[layer] + neg_h[layer], vector)
        extracted.append(
            LayerVector(
                layer=int(layer),
                vector=vector,
                tau=tau,
                n_pos=len(pos_h[layer]),
                n_neg=len(neg_h[layer]),
            )
        )
    return SycophancyVectors(model=model_id, layers=extracted)

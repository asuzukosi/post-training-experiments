"""apply a sycophancy vector: translate (α) or cap excess projection.

steer: h ← h + α v
cap:   h′ = h − v · max(⟨h, v⟩ − τ, 0)
       (ceiling at τ; spec typeset min() which would be a floor)
α ∈ {-2,-1,0,1,2}. v is unit. extraction used last prompt token;
application adds to the residual stream at that layer for all positions.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _tensor(x: Any):
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)


def steer_hidden(hidden: Any, vector: Any, *, alpha: float) -> Any:
    """h + α v, broadcast over leading dims."""
    h = _tensor(hidden)
    v = _tensor(vector)
    return h + float(alpha) * v


def cap_hidden(hidden: Any, vector: Any, *, tau: float) -> Any:
    """subtract excess projection above tau along v."""
    h = _tensor(hidden)
    v = _tensor(vector)
    proj = (h * v).sum(dim=-1, keepdim=True)
    excess = (proj - float(tau)).clamp(min=0)
    return h - excess * v


def parse_alphas(raw: Sequence[float] | str | None = None) -> list[float]:
    if raw is None:
        return list(DEFAULT_ALPHAS)
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(a) for a in raw]
    if not values:
        raise ValueError("alphas must be non-empty")
    return values


def register_residual_hook(
    model: Any,
    *,
    layer: int,
    vector: Sequence[float],
    alpha: float = 0.0,
    tau: float | None = None,
):
    """hook decoder layer `layer` residual; returns the handle (call .remove())."""
    import torch

    v = torch.tensor(list(vector), dtype=torch.float32)
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer {layer} out of range (n={len(layers)})")

    def _hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        v_h = v.to(device=h.device, dtype=h.dtype)
        if tau is not None:
            steered = cap_hidden(h, v_h, tau=tau)
        else:
            steered = steer_hidden(h, v_h, alpha=alpha)
        steered = steered.to(dtype=h.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return layers[layer].register_forward_hook(_hook)


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer: int,
    vector: Sequence[float],
    alpha: float = 0.0,
    tau: float | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    **gen_kwargs: Any,
) -> str:
    """hf generate with a residual hook at `layer`. not vllm — hooks cannot attach there."""
    handle = register_residual_hook(
        model, layer=layer, vector=vector, alpha=alpha, tau=tau
    )
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        do_sample = float(temperature) > 0
        out = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            **gen_kwargs,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()

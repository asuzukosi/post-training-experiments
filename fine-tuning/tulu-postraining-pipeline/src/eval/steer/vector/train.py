"""training and applying the vector, on top of the `steering-vectors` library.

we previously hand-rolled extraction (mean-difference over last-token hiddens) and
application (a residual forward hook). the library does both, and does them better:

    aggregators   mean, PCA, logistic — PCA is more robust when the contrastive pairs
                  are noisy, which ours will be
    operators     addition, ablation, ablation-then-addition. ablation removes the
                  existing component along the direction before adding, which behaves
                  better at large multipliers than pure addition
    layers        every layer in one pass, rather than one at a time

what stays ours: the choice of layers, the on-disk format, and the flip-rate
measurement — the library steers, it does not evaluate.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.steer.vector.caa import to_training_samples
from prepare.paths import resolve_path

DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def middle_layer_ids(n_layers: int) -> list[int]:
    """the middle third, [n/3, 2n/3).

    steering is usually weak in early layers (still lexical) and too late to change the
    output in the last few. never returns empty, even for a tiny model.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    lo = n_layers // 3
    hi = max(lo + 1, (2 * n_layers) // 3)
    return list(range(lo, hi))


@dataclass
class SycophancyVector:
    """a trained steering vector plus what it was trained from."""

    model: str
    layers: list[int]
    aggregator: str
    n_pairs: int
    vector: Any  # steering_vectors.SteeringVector

    @contextmanager
    def applied(self, model: Any, *, alpha: float, ablate: bool = False):
        """patch `model` for the duration of the block.

        `ablate` removes the existing component along the direction before adding it
        back at `alpha`, which holds together better at large |alpha| than pure
        addition. alpha=0 with ablate=False is a genuine no-op.
        """
        from steering_vectors import addition_operator, ablation_then_addition_operator

        operator = ablation_then_addition_operator() if ablate else addition_operator()
        with self.vector.apply(model, multiplier=float(alpha), operator=operator):
            yield


def train_sycophancy_vector(
    rows: Sequence[Any],
    *,
    model: Any,
    tokenizer: Any,
    layers: Sequence[int] | None = None,
    aggregator: str = "pca",
    model_id: str = "",
    batch_size: int = 1,
) -> SycophancyVector:
    """train a steering vector from caa sycophancy rows.

    reads the activation at the LAST prompt token (`read_token_index=-1`), which is the
    convention the contrastive-activation-addition work uses and what our extraction did.

    `aggregator` defaults to pca rather than mean: the mean difference is the naive
    estimator and is dragged around by any pair whose two prompts differ in more than
    the trait.
    """
    from steering_vectors import mean_aggregator, pca_aggregator, train_steering_vector

    aggregators = {"pca": pca_aggregator, "mean": mean_aggregator}
    if aggregator not in aggregators:
        raise ValueError(f"aggregator must be one of {sorted(aggregators)}, got {aggregator!r}")

    samples = to_training_samples(rows)
    layer_ids = (
        list(layers)
        if layers is not None
        else middle_layer_ids(int(model.config.num_hidden_layers))
    )
    vector = train_steering_vector(
        model,
        tokenizer,
        samples,
        layers=layer_ids,
        read_token_index=-1,
        aggregator=aggregators[aggregator](),
        batch_size=batch_size,
        show_progress=True,
    )
    print(
        f"steering vector: {len(samples)} pairs, layers {layer_ids}, aggregator={aggregator}"
    )
    return SycophancyVector(
        model=model_id,
        layers=layer_ids,
        aggregator=aggregator,
        n_pairs=len(samples),
        vector=vector,
    )


def save_vector(vec: SycophancyVector, path: str | Path) -> Path:
    """write the vector as json: metadata plus one array per layer."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": vec.model,
        "layers": vec.layers,
        "aggregator": vec.aggregator,
        "n_pairs": vec.n_pairs,
        "layer_type": vec.vector.layer_type,
        "layer_activations": {
            str(layer): tensor.detach().float().cpu().tolist()
            for layer, tensor in vec.vector.layer_activations.items()
        },
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    print(f"steering vector: layers={len(vec.layers)} wrote={out}")
    return out


def load_vector(path: str | Path) -> SycophancyVector:
    import torch
    from steering_vectors import SteeringVector

    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    activations = {
        int(layer): torch.tensor(values, dtype=torch.float32)
        for layer, values in payload["layer_activations"].items()
    }
    return SycophancyVector(
        model=str(payload.get("model") or ""),
        layers=[int(x) for x in payload["layers"]],
        aggregator=str(payload.get("aggregator") or "pca"),
        n_pairs=int(payload.get("n_pairs") or 0),
        vector=SteeringVector(
            layer_activations=activations,
            layer_type=payload.get("layer_type", "decoder_block"),
        ),
    )


def parse_alphas(raw: Sequence[float] | str | None = None) -> list[float]:
    """the alpha sweep, from a sequence or a comma-separated string."""
    if raw is None:
        return list(DEFAULT_ALPHAS)
    values = (
        [float(part.strip()) for part in raw.split(",") if part.strip()]
        if isinstance(raw, str)
        else [float(a) for a in raw]
    )
    if not values:
        raise ValueError("alphas must be non-empty")
    return values


DEFAULT_GENERATE_BATCH = 16


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    vec: SycophancyVector,
    *,
    alpha: float,
    ablate: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = DEFAULT_GENERATE_BATCH,
    **gen_kwargs: Any,
) -> list[str]:
    """hf generate with the vector applied — one completion per prompt, in batches.

    not vllm: patching needs the live module, so the batch is the only lever on
    throughput here. the flip-rate sweep is 64,260 generations (1,071 probes x 3 repeats
    x 2 turns x 5 alphas x 2 operators), which is ~46 gpu-hours one at a time and ~3
    batched — the largest single line in the programme either way.

    two things this has to get right:

    LEFT PADDING. decoder-only models must be left-padded to batch. right padding puts
    pad tokens between the prompt and the first generated position, so every prompt
    shorter than the longest one in the batch continues from padding instead of from its
    own text — which produces fluent, plausible, unrelated output.

    COMPLETIONS ONLY. `generate` returns the prompt tokens too. returning them would feed
    the probe's own option list back into `chosen_letter`, whose parenthesised-option
    fallback takes the LAST match — so a completion that named no option would score as
    whichever letter the prompt listed last.
    """
    import torch

    from data_tools.chat import ensure_pad_token

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    texts = list(prompts)
    if not texts:
        return []

    ensure_pad_token(tokenizer)
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    do_sample = float(temperature) > 0
    out_texts: list[str] = []
    try:
        with vec.applied(model, alpha=alpha, ablate=ablate):
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size], return_tensors="pt", padding=True
                )
                encoded = {k: v.to(model.device) for k, v in encoded.items()}
                with torch.no_grad():
                    out = model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=do_sample,
                        temperature=float(temperature) if do_sample else None,
                        pad_token_id=tokenizer.pad_token_id,
                        **gen_kwargs,
                    )
                new_tokens = out[:, encoded["input_ids"].shape[1] :]
                out_texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = previous_side
    return out_texts

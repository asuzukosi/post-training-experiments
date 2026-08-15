"""sycophancy steering vectors, on top of the `steering-vectors` library.

we previously hand-rolled extraction (mean-difference over last-token hiddens) and
application (a residual forward hook). the library does both, and does them better:

    aggregators   mean, PCA, logistic — PCA is more robust when the contrastive pairs
                  are noisy, which ours will be
    operators     addition, ablation, ablation-then-addition. ablation removes the
                  existing component along the direction before adding, which behaves
                  better at large multipliers than pure addition
    layers        every layer in one pass, rather than one at a time
    tokens        `min_token_index` / `token_indices` to choose what gets patched

training data is the contrastive activation addition sycophancy set, which is what the
library trains on in its own sycophancy example rather than pairs we author ourselves.

what stays ours: the choice of layers, the on-disk vector format, and the flip-rate
measurement in `flip_rate.py` — the library steers, it does not evaluate.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prepare.paths import resolve_path

DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)

# the contrastive activation addition sycophancy set — 1,000 bio-plus-opinion prompts
# with the sycophantic answer labelled. this is what the steering-vectors library trains
# on in its own sycophancy example, so the pairing is exercised upstream.
CAA_SYCOPHANCY_URL = (
    "https://raw.githubusercontent.com/nrimsky/CAA/main/"
    "datasets/generate/sycophancy/generate_dataset.json"
)
CAA_CACHE = "data/raw/caa_sycophancy_generate.json"
MATCHING_KEY = "answer_matching_behavior"
NOT_MATCHING_KEY = "answer_not_matching_behavior"


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


def require_caa_row(row: Any, *, index: int | None = None) -> dict[str, str]:
    """a caa row: one prompt plus the two answers that differ only in the trait."""
    loc = f" at row {index}" if index is not None else ""
    for key in ("question", MATCHING_KEY, NOT_MATCHING_KEY):
        val = row.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"caa row {key!r} must be a non-empty str{loc}")
    if row[MATCHING_KEY].strip() == row[NOT_MATCHING_KEY].strip():
        raise ValueError(f"caa row has identical answers{loc}; no contrast to learn")
    return {
        "question": row["question"],
        MATCHING_KEY: row[MATCHING_KEY],
        NOT_MATCHING_KEY: row[NOT_MATCHING_KEY],
    }


def to_training_samples(rows: Sequence[Any]) -> list[tuple[str, str]]:
    """(sycophantic, non-sycophantic) prompt pairs for `train_steering_vector`.

    each pair is the SAME question with a different answer appended, so the two prompts
    differ only in the final token. that is the whole point of the caa construction:
    hand-written pairs of two different sentences also differ in topic, wording and
    length, and the vector absorbs all of it.

    DIRECTION: positive is the sycophantic answer, so +alpha steers TOWARDS sycophancy
    and -alpha away from it. the experiment asks whether negative alpha reduces the flip
    rate; getting this backwards would look exactly like steering that does not work.
    """
    out: list[tuple[str, str]] = []
    for i, row in enumerate(rows, start=1):
        r = require_caa_row(row, index=i)
        q = r["question"].rstrip()
        out.append((f"{q}\n{r[MATCHING_KEY].strip()}", f"{q}\n{r[NOT_MATCHING_KEY].strip()}"))
    return out


def load_caa_sycophancy(path: str | Path | None = None) -> list[dict[str, str]]:
    """load the caa sycophancy set, downloading and caching it on first use."""
    import urllib.request

    target = resolve_path(path or CAA_CACHE)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading caa sycophancy set -> {target}")
        urllib.request.urlretrieve(CAA_SYCOPHANCY_URL, target)
    rows = json.loads(target.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"no caa rows in {target}")
    return [require_caa_row(r, index=i) for i, r in enumerate(rows, start=1)]


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


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vec: SycophancyVector,
    *,
    alpha: float,
    ablate: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    **gen_kwargs: Any,
) -> str:
    """hf generate with the vector applied. not vllm — patching needs the live module."""
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    do_sample = float(temperature) > 0
    with vec.applied(model, alpha=alpha, ablate=ablate):
        out = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            **gen_kwargs,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

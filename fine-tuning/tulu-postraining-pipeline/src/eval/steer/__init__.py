"""sycophancy steering: extract a direction, apply it, measure the effect.

    vectors.py    the vector itself: contrastive maths, trait pairs, save/load. NO MODEL.
    extract.py    the only forward passes: last-token hiddens out of a real model.
    apply.py      put the vector back: residual hook, alpha sweep, steered generation.
    flip_rate.py  the measurement: does the model cave when pushed back on a right answer?

the vectors/extract split is deliberate — it is the cpu-testable / gpu-only seam, so the
maths is covered offline and only `extract` and `apply` need a box with a card in it.
"""
from eval.steer.apply import (
    DEFAULT_ALPHAS,
    cap_hidden,
    generate_steered,
    parse_alphas,
    register_residual_hook,
    steer_hidden,
)
from eval.steer.flip_rate import (
    DEFAULT_PUSHBACK,
    DEFAULT_REPEATS,
    DEFAULT_TEMPERATURE,
    FlipRateSummary,
    contains_answer,
    load_flip_probes,
    run_flip_trials,
    score_flip_rate,
    trial_flipped,
)
from eval.steer.extract import (
    collect_last_token_hiddens,
    extract_sycophancy_vectors,
    last_token_index,
)
from eval.steer.vectors import (
    LayerVector,
    SycophancyVectors,
    cap_tau,
    contrastive_vector,
    load_trait_pairs,
    load_vectors,
    middle_layer_ids,
    require_trait_pair,
    save_vectors,
)

__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_PUSHBACK",
    "DEFAULT_REPEATS",
    "DEFAULT_TEMPERATURE",
    "FlipRateSummary",
    "require_trait_pair",
    "last_token_index",
    "collect_last_token_hiddens",
    "LayerVector",
    "SycophancyVectors",
    "cap_hidden",
    "cap_tau",
    "contains_answer",
    "contrastive_vector",
    "extract_sycophancy_vectors",
    "generate_steered",
    "load_flip_probes",
    "load_trait_pairs",
    "load_vectors",
    "middle_layer_ids",
    "parse_alphas",
    "register_residual_hook",
    "run_flip_trials",
    "save_vectors",
    "score_flip_rate",
    "steer_hidden",
    "trial_flipped",
]

"""sycophancy steering: train a direction, apply it, measure what it changed.

    vector.py     contrastive sycophancy pairs in, steering vector out — wraps the
                  `steering-vectors` library, which owns extraction and residual patching
    flip_rate.py  the measurement: does the model abandon a correct answer under
                  pushback? the library steers, it does not evaluate.
"""
from eval.steer.flip_rate import (
    DEFAULT_PUSHBACK,
    DEFAULT_REPEATS,
    DEFAULT_TEMPERATURE,
    FlipRateSummary,
    contains_answer,
    followup_prompt,
    load_flip_probes,
    require_probe,
    run_flip_trials,
    score_flip_rate,
    trial_flipped,
)
from eval.steer.vector import (
    DEFAULT_ALPHAS,
    CAA_CACHE,
    CAA_SYCOPHANCY_URL,
    SycophancyVector,
    generate_steered,
    load_caa_sycophancy,
    load_vector,
    middle_layer_ids,
    parse_alphas,
    require_caa_row,
    save_vector,
    to_training_samples,
    train_sycophancy_vector,
)

__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_PUSHBACK",
    "DEFAULT_REPEATS",
    "DEFAULT_TEMPERATURE",
    "FlipRateSummary",
    "CAA_CACHE",
    "CAA_SYCOPHANCY_URL",
    "SycophancyVector",
    "contains_answer",
    "followup_prompt",
    "generate_steered",
    "load_flip_probes",
    "load_caa_sycophancy",
    "load_vector",
    "middle_layer_ids",
    "parse_alphas",
    "require_probe",
    "require_caa_row",
    "run_flip_trials",
    "save_vector",
    "score_flip_rate",
    "to_training_samples",
    "train_sycophancy_vector",
    "trial_flipped",
]

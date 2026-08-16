"""sycophancy steering vectors.

    caa.py     the contrastive pairs the vector is trained from
    train.py   training, applying and persisting the vector, via `steering-vectors`
"""
from eval.steer.vector.caa import (
    CAA_CACHE,
    CAA_SYCOPHANCY_URL,
    MATCHING_KEY,
    NOT_MATCHING_KEY,
    load_caa_sycophancy,
    require_caa_row,
    to_training_samples,
)
from eval.steer.vector.train import (
    DEFAULT_ALPHAS,
    SycophancyVector,
    generate_steered,
    load_vector,
    middle_layer_ids,
    parse_alphas,
    save_vector,
    train_sycophancy_vector,
)

__all__ = [
    "CAA_CACHE",
    "CAA_SYCOPHANCY_URL",
    "DEFAULT_ALPHAS",
    "MATCHING_KEY",
    "NOT_MATCHING_KEY",
    "SycophancyVector",
    "generate_steered",
    "load_caa_sycophancy",
    "load_vector",
    "middle_layer_ids",
    "parse_alphas",
    "require_caa_row",
    "save_vector",
    "to_training_samples",
    "train_sycophancy_vector",
]

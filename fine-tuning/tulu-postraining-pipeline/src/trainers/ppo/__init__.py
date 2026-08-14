"""ppo training: align sft policy against rm, kl/clip/eos, step ckpts, hub push.

trl 0.19 PPOTrainer.train() does not accept resume_from_checkpoint; if a step
checkpoint exists under the run dir we re-init the policy from it (weights only).

    data.py     prompt pool: load, render + tokenize, hold out a sampling split
    models.py   checkpoint resolution and the four models a ppo step carries
    trainer.py  PPOConfig/PPOTrainer construction and the run entry point
"""
from trainers.ppo.data import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_PROMPTS,
    DEFAULT_NUM_SAMPLE_GENERATIONS,
    load_ppo_prompts,
    ppo_eval_batch_size,
    split_ppo_eval,
    tokenize_ppo_prompts,
)
from trainers.ppo.models import (
    load_ppo_models,
    resolve_rm_checkpoint,
    resolve_sft_checkpoint,
)
from trainers.ppo.trainer import (
    DEFAULT_MISSING_EOS_PENALTY,
    DEFAULT_WANDB_PROJECT,
    build_ppo_config,
    build_ppo_trainer,
    run_ppo,
)

__all__ = [
    "DEFAULT_EVAL_BATCH_SIZE",
    "DEFAULT_EVAL_PROMPTS",
    "DEFAULT_MISSING_EOS_PENALTY",
    "DEFAULT_NUM_SAMPLE_GENERATIONS",
    "DEFAULT_WANDB_PROJECT",
    "build_ppo_config",
    "build_ppo_trainer",
    "load_ppo_models",
    "load_ppo_prompts",
    "ppo_eval_batch_size",
    "resolve_rm_checkpoint",
    "resolve_sft_checkpoint",
    "run_ppo",
    "split_ppo_eval",
    "tokenize_ppo_prompts",
]

"""ppo smoke on a tiny prompt subset."""
from __future__ import annotations

import os

from pathlib import Path

import pytest

from trainers.ppo import (
    DEFAULT_MISSING_EOS_PENALTY,
    build_ppo_config,
    resolve_rm_checkpoint,
    resolve_sft_checkpoint,
)

PPO_OVERRIDES = {
    "num_train_epochs": 1,
    "total_episodes": 8,
    "batch_size": 2,
    "mini_batch_size": 1,
    "per_device_train_batch_size": 2,
    "num_mini_batches": 1,
    "ppo_epochs": 1,
    "max_prompt_length": 64,
    "max_new_tokens": 16,
    "score_eos_only": True,
    "kl_coef": 0.05,
    "cliprange": 0.2,
}


def test_build_ppo_config_maps_aliases(tmp_path: Path, smoke_cfg) -> None:
    args = build_ppo_config(
        smoke_cfg("ppo", **PPO_OVERRIDES),
        run_name="ppo_smoke_cfg",
        output_dir=tmp_path,
        push_to_hub=False,
    )
    assert args.kl_coef == 0.05
    assert args.cliprange == 0.2
    assert args.num_ppo_epochs == 1
    assert args.response_length == 16
    assert args.stop_token == "eos"
    assert args.missing_eos_penalty == DEFAULT_MISSING_EOS_PENALTY
    assert args.per_device_train_batch_size == 2
    assert args.push_to_hub is False



@pytest.mark.skipif(
    os.environ.get("RUN_PPO_SMOKE") != "1",
    reason="set RUN_PPO_SMOKE=1 to run the gpu/model smoke",
)
def test_ppo_smoke_train_tiny_subset(
    tmp_path: Path, smoke_cfg, smoke_dataset, assert_saved_model, require_env
) -> None:
    from trainers.ppo import run_ppo

    sft_ckpt, rm_ckpt = require_env(
        "PPO_SMOKE_SFT_CHECKPOINT", "PPO_SMOKE_RM_CHECKPOINT"
    )
    out = run_ppo(
        smoke_cfg("ppo", **PPO_OVERRIDES),
        sft_checkpoint=sft_ckpt,
        rm_checkpoint=rm_ckpt,
        dataset=smoke_dataset("ppo"),
        run_name="ppo_smoke",
        output_dir=tmp_path / "ppo_smoke",
        push_to_hub=False,
    )
    assert_saved_model(out)

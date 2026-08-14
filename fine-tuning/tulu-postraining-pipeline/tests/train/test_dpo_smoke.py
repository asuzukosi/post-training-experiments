"""dpo smoke on a tiny preference-pair subset."""
from __future__ import annotations

import os

from pathlib import Path

import pytest

from trainers.dpo import (
    build_dpo_config,
    format_dpo_task,
    resolve_beta,
    resolve_sft_checkpoint,
)

DPO_OVERRIDES = {"num_train_epochs": 1, "precompute_ref_log_probs": True}
SMOKE_BETA = 0.05



def test_resolve_beta_requires_or_validates() -> None:
    cfg = {"betas": [0.05, 0.1]}
    with pytest.raises(ValueError, match="beta is required"):
        resolve_beta(cfg, None)
    with pytest.raises(ValueError, match="not in configs betas"):
        resolve_beta(cfg, 0.2)
    assert resolve_beta(cfg, 0.05) == 0.05
    assert resolve_beta({"betas": [0.1]}, None) == 0.1


def test_build_dpo_config_sets_beta(tmp_path: Path, smoke_cfg) -> None:
    args = build_dpo_config(
        smoke_cfg("dpo", **DPO_OVERRIDES),
        beta=SMOKE_BETA,
        run_name="dpo_smoke_cfg",
        output_dir=tmp_path,
        push_to_hub=False,
    )
    assert args.beta == SMOKE_BETA
    assert args.precompute_ref_log_probs is True
    assert args.max_steps == 2
    assert args.push_to_hub is False



@pytest.mark.skipif(
    os.environ.get("RUN_DPO_SMOKE") != "1",
    reason="set RUN_DPO_SMOKE=1 to run the gpu/model smoke",
)
def test_dpo_smoke_train_tiny_subset(
    tmp_path: Path, smoke_cfg, smoke_dataset, assert_saved_model, assert_trained, require_env
) -> None:
    from trainers.dpo import run_dpo

    (sft_ckpt,) = require_env("DPO_SMOKE_SFT_CHECKPOINT")
    out = run_dpo(
        smoke_cfg("dpo", **DPO_OVERRIDES),
        beta=SMOKE_BETA,
        sft_checkpoint=sft_ckpt,
        dataset=smoke_dataset("dpo"),
        run_name="dpo_smoke",
        output_dir=tmp_path / "dpo_smoke",
        push_to_hub=False,
    )
    assert_saved_model(out)
    assert_trained(out)

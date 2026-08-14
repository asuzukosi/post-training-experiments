"""rm smoke on a tiny preference-pair subset."""
from __future__ import annotations

import os

from pathlib import Path

import pytest

from trainers.rm import RM_NUM_TRAIN_EPOCHS, build_rm_config, resolve_sft_checkpoint

# 99 is deliberate: the rm trainer must force this back to 1 epoch
RM_OVERRIDES = {"num_train_epochs": 99}


def test_build_rm_config_hardcodes_one_epoch(tmp_path: Path, smoke_cfg) -> None:
    args = build_rm_config(
        smoke_cfg("rm", **RM_OVERRIDES),
        run_name="rm_smoke_cfg",
        output_dir=tmp_path,
        push_to_hub=False,
    )
    assert args.num_train_epochs == float(RM_NUM_TRAIN_EPOCHS)
    assert args.max_steps == 2
    assert args.push_to_hub is False
    assert args.report_to in ([], "none", ["none"])



@pytest.mark.skipif(
    os.environ.get("RUN_RM_SMOKE") != "1",
    reason="set RUN_RM_SMOKE=1 to run the gpu/model smoke",
)
def test_rm_smoke_train_tiny_subset(
    tmp_path: Path, smoke_cfg, smoke_dataset, assert_saved_model, require_env
) -> None:
    from trainers.rm import run_rm

    (sft_ckpt,) = require_env("RM_SMOKE_SFT_CHECKPOINT")
    out = run_rm(
        smoke_cfg("rm", **RM_OVERRIDES),
        sft_checkpoint=sft_ckpt,
        dataset=smoke_dataset("rm"),
        run_name="rm_smoke",
        output_dir=tmp_path / "rm_smoke",
        push_to_hub=False,
    )
    assert_saved_model(out)

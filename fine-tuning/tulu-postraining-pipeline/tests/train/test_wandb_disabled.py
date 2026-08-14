"""a smoke must never phone home to w&b.

trainers read `use_wandb` via cfg_use_wandb and derive hf report_to from that.
smokes set use_wandb: false in conftest overrides.
"""
from __future__ import annotations

from trainers.rm import build_rm_config


def test_smoke_overrides_disable_wandb(tmp_path, smoke_cfg) -> None:
    args = build_rm_config(
        smoke_cfg("rm"), run_name="wandb_off", output_dir=tmp_path, push_to_hub=False
    )
    assert args.report_to in ([], "none", ["none"])

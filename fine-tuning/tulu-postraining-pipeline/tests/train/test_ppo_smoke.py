"""ppo smoke on a tiny prompt subset.

phase 2: code only. phase 5 gpu run:
  cd fine-tuning/tulu-postraining-pipeline
  RUN_PPO_SMOKE=1 PPO_SMOKE_SFT_CHECKPOINT=<sft> PPO_SMOKE_RM_CHECKPOINT=<rm> \\
    PYTHONPATH=src python -m pytest tests/train/test_ppo_smoke.py -s
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from trainers.ppo import (
    DEFAULT_MISSING_EOS_PENALTY,
    build_ppo_config,
    resolve_rm_checkpoint,
    resolve_sft_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_prompts(n: int = 8) -> list[dict]:
    return [
        {"prompt": f"say hello number {i}", "prompt_id": f"smoke-{i}"}
        for i in range(n)
    ]


def _smoke_cfg() -> dict:
    with (REPO_ROOT / "configs" / "ppo.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        {
            "num_train_epochs": 1,
            "total_episodes": 8,
            "batch_size": 2,
            "mini_batch_size": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_mini_batches": 1,
            "ppo_epochs": 1,
            "max_prompt_length": 64,
            "max_new_tokens": 16,
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "score_eos_only": True,
            "kl_coef": 0.05,
            "cliprange": 0.2,
        }
    )
    return cfg


def test_build_ppo_config_maps_aliases(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    args = build_ppo_config(
        cfg,
        run_name="ppo_smoke_cfg",
        output_dir=tmp_path / "out",
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


def test_resolve_checkpoints_require_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sft_checkpoint is required"):
        resolve_sft_checkpoint({"sft_checkpoint": None}, None)
    with pytest.raises(ValueError, match="rm_checkpoint is required"):
        resolve_rm_checkpoint({"rm_checkpoint": None}, None)

    sft = tmp_path / "sft"
    rm = tmp_path / "rm"
    sft.mkdir()
    rm.mkdir()
    assert resolve_sft_checkpoint({"sft_checkpoint": str(sft)}, None) == sft.resolve()
    assert resolve_rm_checkpoint({"rm_checkpoint": str(rm)}, None) == rm.resolve()


@pytest.mark.skipif(
    os.environ.get("RUN_PPO_SMOKE") != "1",
    reason="set RUN_PPO_SMOKE=1 to run the gpu/model smoke (phase 5)",
)
def test_ppo_smoke_train_tiny_subset(tmp_path: Path) -> None:
    from datasets import Dataset

    from trainers.ppo import run_ppo

    sft_ckpt = os.environ.get("PPO_SMOKE_SFT_CHECKPOINT")
    rm_ckpt = os.environ.get("PPO_SMOKE_RM_CHECKPOINT")
    if not sft_ckpt or not rm_ckpt:
        pytest.skip("set PPO_SMOKE_SFT_CHECKPOINT and PPO_SMOKE_RM_CHECKPOINT")

    cfg = _smoke_cfg()
    ds = Dataset.from_list(_tiny_prompts(8))
    out = run_ppo(
        cfg,
        sft_checkpoint=sft_ckpt,
        rm_checkpoint=rm_ckpt,
        dataset=ds,
        run_name="ppo_smoke",
        output_dir=tmp_path / "ppo_smoke",
        push_to_hub=False,
    )
    assert out.is_dir()
    assert (out / "config.json").is_file()
    assert any(out.glob("*.safetensors")) or (out / "model.safetensors.index.json").is_file()

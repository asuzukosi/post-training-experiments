"""rm smoke on a tiny preference-pair subset.

phase 2: code only. phase 5 gpu run:
  cd fine-tuning/tulu-postraining-pipeline
  RUN_RM_SMOKE=1 PYTHONPATH=src python -m pytest tests/train/test_rm_smoke.py -s
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from trainers.rm import RM_NUM_TRAIN_EPOCHS, build_rm_config, resolve_sft_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_pairs(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        prompt = f"question {i}: what is {i}+{i}?"
        rows.append(
            {
                "prompt": prompt,
                "prompt_id": f"smoke-{i}",
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"correct: {i + i}"},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "i do not know"},
                ],
                "score_chosen": 8.0,
                "score_rejected": 3.0,
            }
        )
    return rows


def _smoke_cfg() -> dict:
    with (REPO_ROOT / "configs" / "rm.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        {
            "num_train_epochs": 99,  # must be ignored / forced to 1
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_length": 256,
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "warmup_ratio": 0.0,
        }
    )
    return cfg


def test_build_rm_config_hardcodes_one_epoch(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    args = build_rm_config(
        cfg,
        run_name="rm_smoke_cfg",
        output_dir=tmp_path / "out",
        push_to_hub=False,
    )
    assert args.num_train_epochs == float(RM_NUM_TRAIN_EPOCHS)
    assert args.max_steps == 2
    assert args.push_to_hub is False
    assert args.report_to in ([], "none", ["none"])


def test_resolve_sft_checkpoint_requires_path(tmp_path: Path) -> None:
    cfg = {"sft_checkpoint": None}
    with pytest.raises(ValueError, match="sft_checkpoint is required"):
        resolve_sft_checkpoint(cfg, None)

    ckpt = tmp_path / "sft_ckpt"
    ckpt.mkdir()
    resolved = resolve_sft_checkpoint({"sft_checkpoint": str(ckpt)}, None)
    assert resolved == ckpt.resolve() or resolved == ckpt


@pytest.mark.skipif(
    os.environ.get("RUN_RM_SMOKE") != "1",
    reason="set RUN_RM_SMOKE=1 to run the gpu/model smoke (phase 5)",
)
def test_rm_smoke_train_tiny_subset(tmp_path: Path) -> None:
    from datasets import Dataset

    from trainers.rm import run_rm

    sft_ckpt = os.environ.get("RM_SMOKE_SFT_CHECKPOINT")
    if not sft_ckpt:
        pytest.skip("set RM_SMOKE_SFT_CHECKPOINT to an sft checkpoint path")

    cfg = _smoke_cfg()
    ds = Dataset.from_list(_tiny_pairs(8))
    out = run_rm(
        cfg,
        sft_checkpoint=sft_ckpt,
        dataset=ds,
        run_name="rm_smoke",
        output_dir=tmp_path / "rm_smoke",
        push_to_hub=False,
    )
    assert out.is_dir()
    assert (out / "config.json").is_file()
    assert any(out.glob("*.safetensors")) or (out / "model.safetensors.index.json").is_file()

"""dpo smoke on a tiny preference-pair subset.

phase 2: code only. phase 5 gpu run:
  cd fine-tuning/tulu-postraining-pipeline
  RUN_DPO_SMOKE=1 DPO_SMOKE_SFT_CHECKPOINT=<sft> \\
    PYTHONPATH=src python -m pytest tests/test_dpo_smoke.py -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.dpo import build_dpo_config, format_dpo_task, resolve_beta, resolve_sft_checkpoint


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
    with (ROOT / "configs" / "dpo.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        {
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_length": 256,
            "max_prompt_length": 128,
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "warmup_ratio": 0.0,
            "precompute_ref_log_probs": True,
        }
    )
    return cfg


def test_format_dpo_task() -> None:
    assert format_dpo_task(0.05) == "dpo-b0.05"
    assert format_dpo_task(0.1) == "dpo-b0.1"


def test_resolve_beta_requires_or_validates() -> None:
    cfg = {"betas": [0.05, 0.1]}
    with pytest.raises(ValueError, match="beta is required"):
        resolve_beta(cfg, None)
    with pytest.raises(ValueError, match="not in configs betas"):
        resolve_beta(cfg, 0.2)
    assert resolve_beta(cfg, 0.05) == 0.05
    assert resolve_beta({"betas": [0.1]}, None) == 0.1


def test_build_dpo_config_sets_beta(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    args = build_dpo_config(
        cfg,
        beta=0.05,
        run_name="dpo_smoke_cfg",
        output_dir=tmp_path / "out",
        push_to_hub=False,
    )
    assert args.beta == 0.05
    assert args.precompute_ref_log_probs is True
    assert args.max_steps == 2
    assert args.push_to_hub is False


def test_resolve_sft_checkpoint_requires_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sft_checkpoint is required"):
        resolve_sft_checkpoint({"sft_checkpoint": None}, None)
    ckpt = tmp_path / "sft_ckpt"
    ckpt.mkdir()
    resolved = resolve_sft_checkpoint({"sft_checkpoint": str(ckpt)}, None)
    assert resolved == ckpt.resolve() or resolved == ckpt


@pytest.mark.skipif(
    os.environ.get("RUN_DPO_SMOKE") != "1",
    reason="set RUN_DPO_SMOKE=1 to run the gpu/model smoke (phase 5)",
)
def test_dpo_smoke_train_tiny_subset(tmp_path: Path) -> None:
    from datasets import Dataset

    from pipeline.dpo import run_dpo

    sft_ckpt = os.environ.get("DPO_SMOKE_SFT_CHECKPOINT")
    if not sft_ckpt:
        pytest.skip("set DPO_SMOKE_SFT_CHECKPOINT to an sft checkpoint path")

    cfg = _smoke_cfg()
    ds = Dataset.from_list(_tiny_pairs(8))
    out = run_dpo(
        cfg,
        beta=0.05,
        sft_checkpoint=sft_ckpt,
        dataset=ds,
        run_name="dpo_smoke",
        output_dir=tmp_path / "dpo_smoke",
        push_to_hub=False,
    )
    assert out.is_dir()
    assert (out / "config.json").is_file()
    assert any(out.glob("*.safetensors")) or (out / "model.safetensors.index.json").is_file()

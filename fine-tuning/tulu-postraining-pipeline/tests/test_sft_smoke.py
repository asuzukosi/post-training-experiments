"""sft smoke on a tiny conversational subset.

phase 2: code only. phase 5 gpu run:
  cd fine-tuning/tulu-postraining-pipeline
  RUN_SFT_SMOKE=1 PYTHONPATH=src python -m pytest tests/test_sft_smoke.py -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.sft import build_sft_config, run_sft


def _tiny_messages(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"smoke-{i}",
                "source": "smoke",
                "messages": [
                    {"role": "user", "content": f"question {i}: what is {i}+{i}?"},
                    {
                        "role": "assistant",
                        "content": f"answer {i}: {i}+{i} equals {i + i}.",
                    },
                ],
            }
        )
    return rows


def _smoke_cfg() -> dict:
    with (ROOT / "configs" / "sft.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        {
            "base_model": os.environ.get("SFT_SMOKE_MODEL", cfg["base_model"]),
            "num_train_epochs": 1,
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_length": 256,
            "packing": False,
            "assistant_only_loss": True,
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "warmup_ratio": 0.0,
        }
    )
    return cfg


def test_build_sft_config_accepts_smoke_overrides(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    args = build_sft_config(
        cfg,
        run_name="sft_smoke_cfg",
        output_dir=tmp_path / "out",
        push_to_hub=False,
    )
    assert args.max_steps == 2
    assert args.packing is False
    assert args.assistant_only_loss is True
    assert args.push_to_hub is False
    # transformers normalizes report_to="none" to []
    assert args.report_to in ([], "none", ["none"])


@pytest.mark.skipif(
    os.environ.get("RUN_SFT_SMOKE") != "1",
    reason="set RUN_SFT_SMOKE=1 to run the gpu/model smoke (phase 5)",
)
def test_sft_smoke_train_tiny_subset(tmp_path: Path) -> None:
    from datasets import Dataset

    cfg = _smoke_cfg()
    ds = Dataset.from_list(_tiny_messages(8))
    out = run_sft(
        cfg,
        dataset=ds,
        run_name="sft_smoke",
        output_dir=tmp_path / "sft_smoke",
        push_to_hub=False,
    )
    assert out.is_dir()
    assert (out / "config.json").is_file()
    assert any(out.glob("*.safetensors")) or (out / "model.safetensors.index.json").is_file()

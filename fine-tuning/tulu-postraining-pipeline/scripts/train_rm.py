#!/usr/bin/env python3
"""train reward model from configs/rm.yaml (init from sft checkpoint).

example runs:
  # prepare pairs first (once)
  python scripts/prepare_data.py --rm

  # full train (auto-resumes latest checkpoint under the run dir if present)
  python scripts/train_rm.py --sft-checkpoint results/checkpoints/<sft_run>
  python scripts/train_rm.py --config configs/rm.yaml --sft-checkpoint <sft_run_or_path>

  # skip hub push (local/debug)
  python scripts/train_rm.py --sft-checkpoint <sft> --no-hub

  # optional overrides
  python scripts/train_rm.py --sft-checkpoint <sft> --run-name qwen2.5-1.5b_rm_debug
  python scripts/train_rm.py --sft-checkpoint <sft> --hub-username <you>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.reward_model import run_rm


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train reward model stage")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rm.yaml",
        help="path to rm yaml config",
    )
    p.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="path to sft checkpoint used to init the rm (required unless set in yaml)",
    )
    p.add_argument("--run-name", type=str, default=None, help="override run name")
    p.add_argument("--hub-username", type=str, default=None, help="hf hub username/org")
    p.add_argument(
        "--no-hub",
        action="store_true",
        help="skip hub push_to_hub / upload",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="override wandb project name",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_rm(
        load_yaml(args.config),
        sft_checkpoint=args.sft_checkpoint,
        run_name=args.run_name,
        hub_username=args.hub_username,
        push_to_hub=not args.no_hub,
        wandb_project=args.wandb_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""train sft from configs/sft.yaml.

example runs:
  # prepare subset first (once)
  python scripts/prepare/sft.py

  # full train (auto-resumes latest checkpoint under the run dir if present)
  python scripts/train/sft.py
  python scripts/train/sft.py --config configs/sft.yaml

  # skip hub push (local/debug)
  python scripts/train/sft.py --no-hub

  # optional overrides
  python scripts/train/sft.py --run-name qwen2.5-1.5b_sft_debug --wandb-project tulu-postraining
  python scripts/train/sft.py --hub-username <you>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trainers.sft import run_sft


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train sft stage")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sft.yaml",
        help="path to sft yaml config",
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
    run_sft(
        load_yaml(args.config),
        run_name=args.run_name,
        hub_username=args.hub_username,
        push_to_hub=not args.no_hub,
        wandb_project=args.wandb_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

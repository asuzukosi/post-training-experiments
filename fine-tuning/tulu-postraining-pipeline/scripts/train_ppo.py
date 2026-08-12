#!/usr/bin/env python3
"""train ppo from configs/ppo.yaml (policy from sft, reward from rm).

example runs:
  # prepare prompts first (once)
  python scripts/prepare_data.py --ppo

  # full train (reloads latest policy weights under the run dir if present)
  python scripts/train_ppo.py \\
    --sft-checkpoint results/checkpoints/<sft_run> \\
    --rm-checkpoint results/checkpoints/<rm_run>

  # skip hub push (local/debug)
  python scripts/train_ppo.py --sft-checkpoint <sft> --rm-checkpoint <rm> --no-hub

  # optional overrides
  python scripts/train_ppo.py --sft-checkpoint <sft> --rm-checkpoint <rm> --run-name qwen2.5-1.5b_ppo_debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.ppo import run_ppo


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train ppo stage")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "ppo.yaml",
        help="path to ppo yaml config",
    )
    p.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="path to sft checkpoint used to init policy+ref (required unless set in yaml)",
    )
    p.add_argument(
        "--rm-checkpoint",
        type=str,
        default=None,
        help="path to rm checkpoint used as reward model (required unless set in yaml)",
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
    run_ppo(
        load_yaml(args.config),
        sft_checkpoint=args.sft_checkpoint,
        rm_checkpoint=args.rm_checkpoint,
        run_name=args.run_name,
        hub_username=args.hub_username,
        push_to_hub=not args.no_hub,
        wandb_project=args.wandb_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

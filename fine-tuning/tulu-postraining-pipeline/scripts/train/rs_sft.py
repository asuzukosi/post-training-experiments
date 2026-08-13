#!/usr/bin/env python3
"""train rs-sft from configs/rs_sft.yaml (reuses trainers.sft).

inits from model_sft, then prompt-masked sft on bon tournament top-1.
data must already exist at cfg.processed_path.

examples:
  python scripts/eval/bon.py --generations results/metrics/rs/gens.jsonl
  python scripts/train/rs_sft.py --sft-checkpoint results/checkpoints/<sft_run>
  python scripts/train/rs_sft.py --sft-checkpoint <sft> --no-hub
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trainers.sft import resolve_sft_checkpoint, run_sft

DEFAULT_CONFIG = ROOT / "configs" / "rs_sft.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train rs-sft stage from model_sft")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="path to model_sft used to init rs-sft (required unless set in yaml)",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--hub-username", type=str, default=None)
    p.add_argument("--no-hub", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    ckpt = resolve_sft_checkpoint(cfg, args.sft_checkpoint)
    cfg["base_model"] = str(ckpt)
    print(f"rs-sft init from sft_checkpoint={ckpt}")
    run_sft(
        cfg,
        run_name=args.run_name,
        hub_username=args.hub_username,
        push_to_hub=not args.no_hub,
        wandb_project=args.wandb_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

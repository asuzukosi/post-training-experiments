#!/usr/bin/env python3
"""run rewardbench-chat gate on an rm checkpoint.

examples:
  python scripts/eval/rb_gate.py --rm-checkpoint results/checkpoints/<rm>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.rb_gate_chat import run_reward_bench_gate
from prepare.paths import resolve_path

DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run rewardbench-chat gate")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--rm-checkpoint",
        type=str,
        required=True,
        help="reward model checkpoint",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_reward_bench_gate(
        args.rm_checkpoint,
        metrics_path=metrics_dir / "rb_gate_chat.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

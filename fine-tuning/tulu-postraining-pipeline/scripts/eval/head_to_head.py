#!/usr/bin/env python3
"""generate a vs b, judge position-swapped, write style reports.

examples:
  python scripts/eval/head_to_head.py \\
    --a results/checkpoints/<sft> \\
    --b results/checkpoints/<dpo> \\
    --prompts data/processed/eval_prompts.jsonl \\
    --runs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.head_to_head import run_head_to_head
from eval.judge import DEFAULT_JUDGE_MODEL
from eval.style import DEFAULT_MAX_REL_LENGTH_DIFF
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
    p = argparse.ArgumentParser(description="run pairwise head-to-head eval")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--a", type=str, required=True, help="model a checkpoint/id")
    p.add_argument("--b", type=str, required=True, help="model b checkpoint/id")
    p.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="jsonl of {id,prompt}",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--runs", type=int, default=None)
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
    runs = int(args.runs if args.runs is not None else cfg.get("default_runs", 3))
    judge_model = args.judge_model or cfg.get("judge_model") or DEFAULT_JUDGE_MODEL
    h2h_dir = metrics_dir / "head_to_head" / f"{Path(args.a).name}_vs_{Path(args.b).name}"
    run_head_to_head(
        model_a=args.a,
        model_b=args.b,
        prompts_path=args.prompts,
        output_dir=h2h_dir,
        runs=runs,
        judge_model=str(judge_model),
        max_rel_length_diff=float(
            cfg.get("max_rel_length_diff", DEFAULT_MAX_REL_LENGTH_DIFF)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

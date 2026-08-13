#!/usr/bin/env python3
"""run lm-eval skills (ifeval / mmlu).

examples:
  python scripts/eval/skills.py --model Qwen/Qwen2.5-1.5B
  python scripts/eval/skills.py --mmlu --model results/checkpoints/<sft> \\
    --baseline-mmlu-acc 0.37
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.lm_eval_skills import DEFAULT_TASKS, run_skills_eval
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
    p = argparse.ArgumentParser(description="run ifeval/mmlu skills eval")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", type=str, required=True, help="model id or checkpoint")
    p.add_argument("--ifeval", action="store_true", help="run ifeval (default: both)")
    p.add_argument("--mmlu", action="store_true", help="run mmlu (default: both)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--baseline-mmlu-acc",
        type=float,
        default=None,
        help="prior mmlu acc in [0,1] for >5pt drop flag",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="optional lm-eval limit (fraction or count) for smoke runs",
    )
    return p.parse_args(argv)


def _tasks(args: argparse.Namespace) -> list[str]:
    if args.ifeval and args.mmlu:
        return list(DEFAULT_TASKS)
    if args.ifeval:
        return ["ifeval"]
    if args.mmlu:
        return ["mmlu"]
    return list(DEFAULT_TASKS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args)
    out = metrics_dir / f"skills_{Path(args.model).name}.json"
    run_skills_eval(
        args.model,
        tasks=tasks,
        output_path=out,
        baseline_mmlu_acc=args.baseline_mmlu_acc,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

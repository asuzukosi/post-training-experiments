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

from eval.lm_eval_skills import DEFAULT_TASK_LIMITS, DEFAULT_TASKS, run_skills_eval
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
        help="global lm-eval limit for smoke runs; per-task limits in eval.yaml win",
    )
    p.add_argument(
        "--task-limit",
        action="append",
        default=None,
        metavar="TASK=N",
        help="per-task depth, e.g. --task-limit mmlu=25 --task-limit ifeval=all "
        "(repeatable; overrides eval.yaml and --limit)",
    )
    return p.parse_args(argv)


def parse_task_limits(raw: list[str] | None) -> dict[str, float | int | None]:
    """`mmlu=25` / `ifeval=all` -> {"mmlu": 25, "ifeval": None}."""
    out: dict[str, float | int | None] = {}
    for item in raw or []:
        task, _, value = str(item).partition("=")
        task = task.strip()
        if not task or not _:
            raise ValueError(f"--task-limit must be TASK=N or TASK=all, got {item!r}")
        text = value.strip().lower()
        if text in ("all", "none", ""):
            out[task] = None
        else:
            out[task] = int(float(text)) if float(text).is_integer() else float(text)
    return out


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
    # eval.yaml declares the depths; --task-limit overrides one without editing config
    task_limits = {**DEFAULT_TASK_LIMITS, **(cfg.get("skills_task_limits") or {})}
    task_limits.update(parse_task_limits(args.task_limit))
    out = metrics_dir / f"skills_{Path(args.model).name}.json"
    run_skills_eval(
        args.model,
        tasks=tasks,
        output_path=out,
        baseline_mmlu_acc=args.baseline_mmlu_acc,
        limit=args.limit,
        task_limits={t: task_limits[t] for t in tasks if t in task_limits},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

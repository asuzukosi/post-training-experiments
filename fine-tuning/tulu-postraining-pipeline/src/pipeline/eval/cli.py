"""cli for skills / reward-bench / head-to-head / style evals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pipeline.eval.head_to_head import run_head_to_head
from pipeline.eval.judge import DEFAULT_JUDGE_MODEL
from pipeline.eval.lm_eval_skills import DEFAULT_TASKS, run_skills_eval
from pipeline.eval.style import (
    DEFAULT_MAX_REL_LENGTH_DIFF,
    report_head_to_head_style_from_jsonl,
)
from pipeline.prepare.paths import ROOT, resolve_path
from pipeline.rb_gate_chat import run_reward_bench_gate

DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "run eval suites: skills (ifeval/mmlu), reward-bench, "
            "head-to-head, or style report from judge jsonl"
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to eval yaml defaults",
    )

    modes = p.add_argument_group("modes (pick one or more)")
    modes.add_argument("--ifeval", action="store_true", help="run lm-eval ifeval")
    modes.add_argument("--mmlu", action="store_true", help="run lm-eval mmlu")
    modes.add_argument(
        "--reward-bench",
        action="store_true",
        help="run rewardbench-chat gate on an rm checkpoint",
    )
    modes.add_argument(
        "--head-to-head",
        action="store_true",
        help="generate a vs b, judge position-swapped, write style reports",
    )
    modes.add_argument(
        "--style",
        action="store_true",
        help="length/markdown + length-controlled win-rate from a judge jsonl",
    )

    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="model/checkpoint for --ifeval/--mmlu",
    )
    p.add_argument(
        "--rm-checkpoint",
        type=str,
        default=None,
        help="reward model checkpoint for --reward-bench",
    )
    p.add_argument(
        "--a",
        type=str,
        default=None,
        help="model a checkpoint/id for --head-to-head",
    )
    p.add_argument(
        "--b",
        type=str,
        default=None,
        help="model b checkpoint/id for --head-to-head",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="jsonl of {id,prompt} for --head-to-head",
    )
    p.add_argument(
        "--judge-jsonl",
        type=Path,
        default=None,
        help="existing judge jsonl for --style",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="metrics output dir (default: configs/eval.yaml metrics_dir)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="in-process vllm judge model (default from config / 32b)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=None,
        help="head-to-head repeats (default from config)",
    )
    p.add_argument(
        "--length-control",
        action="store_true",
        help="emphasize length-controlled metrics in head-to-head/style",
    )
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
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="pass prompts to generators as-is (skip apply_chat_template)",
    )
    return p.parse_args(argv)


def _selected_modes(args: argparse.Namespace) -> list[str]:
    modes = []
    if args.ifeval or args.mmlu:
        modes.append("skills")
    if args.reward_bench:
        modes.append("reward_bench")
    if args.head_to_head:
        modes.append("head_to_head")
    if args.style:
        modes.append("style")
    return modes


def _skills_tasks(args: argparse.Namespace) -> list[str]:
    if args.ifeval and args.mmlu:
        return list(DEFAULT_TASKS)
    if args.ifeval:
        return ["ifeval"]
    if args.mmlu:
        return ["mmlu"]
    return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    modes = _selected_modes(args)
    if not modes:
        print(
            "error: select at least one mode: "
            "--ifeval/--mmlu/--reward-bench/--head-to-head/--style",
            file=sys.stderr,
        )
        return 2

    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if "skills" in modes:
        if not args.model:
            print("error: --model is required for --ifeval/--mmlu", file=sys.stderr)
            return 2
        tasks = _skills_tasks(args)
        out = metrics_dir / f"skills_{Path(args.model).name}.json"
        run_skills_eval(
            args.model,
            tasks=tasks,
            model_backend=str(cfg.get("model_backend", "hf")),
            output_path=out,
            baseline_mmlu_acc=args.baseline_mmlu_acc,
            limit=args.limit,
        )

    if "reward_bench" in modes:
        if not args.rm_checkpoint:
            print(
                "error: --rm-checkpoint is required for --reward-bench",
                file=sys.stderr,
            )
            return 2
        run_reward_bench_gate(
            args.rm_checkpoint,
            metrics_path=metrics_dir / "rb_gate_chat.json",
        )

    if "head_to_head" in modes:
        if not args.a or not args.b or args.prompts is None:
            print(
                "error: --head-to-head needs --a --b --prompts",
                file=sys.stderr,
            )
            return 2
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
            apply_chat_template=not args.no_chat_template
            and bool(cfg.get("apply_chat_template", True)),
            length_control=args.length_control,
            max_rel_length_diff=float(
                cfg.get("max_rel_length_diff", DEFAULT_MAX_REL_LENGTH_DIFF)
            ),
        )

    if "style" in modes:
        if args.judge_jsonl is None:
            print("error: --style needs --judge-jsonl", file=sys.stderr)
            return 2
        report = report_head_to_head_style_from_jsonl(
            args.judge_jsonl,
            max_rel_length_diff=float(
                cfg.get("max_rel_length_diff", DEFAULT_MAX_REL_LENGTH_DIFF)
            ),
        )
        out = metrics_dir / f"style_{Path(args.judge_jsonl).stem}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
            f.write("\n")
        print(
            f"style: raw_win_b={report.raw.win_rate_b} "
            f"lc_win_b={report.length_controlled.win_rate_b} wrote={out}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

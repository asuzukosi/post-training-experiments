#!/usr/bin/env python3
"""write rs-sft vs dpo verdict from a direct head-to-head summary.

model_b in the summary must be the rs checkpoint. claim is teacher-distill
plus selection vs dpo on that teacher's preferences, not rs-in-general.

examples:
  python scripts/analysis/rs_verdict.py \\
    --summary results/metrics/head_to_head/.../summary_dpo_vs_rs.json \\
    --judge-bias results/metrics/judge_bias_....json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.io import DEFAULT_METRICS_DIR, load_json_mapping
from analysis.verdict import (
    arm_from_head_to_head_summary,
    build_rs_dpo_verdict,
    write_rs_dpo_verdict,
)
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="write rs-sft vs dpo verdict")
    p.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="head-to-head summary json (a=dpo, b=rs)",
    )
    p.add_argument("--rs-name", type=str, default="rs_sft")
    p.add_argument("--dpo-name", type=str, default="dpo")
    p.add_argument(
        "--judge-bias",
        type=Path,
        default=None,
        help="optional judge_bias json from scripts/eval/judge_bias.py",
    )
    p.add_argument("--metrics-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        rs = arm_from_head_to_head_summary(args.rs_name, args.summary)
        bias = load_json_mapping(args.judge_bias) if args.judge_bias is not None else None
        verdict = build_rs_dpo_verdict(rs, dpo_name=args.dpo_name, judge_bias=bias)
        write_rs_dpo_verdict(
            verdict,
            metrics_dir / "rs_sft_vs_dpo_verdict.json",
            markdown_path=metrics_dir / "rs_sft_vs_dpo_verdict.md",
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

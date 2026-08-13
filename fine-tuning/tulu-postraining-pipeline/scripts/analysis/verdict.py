#!/usr/bin/env python3
"""write dpo-vs-ppo equal-data verdict from vs-sft summaries.

examples:
  python scripts/analysis/verdict.py \\
    --dpo-name dpo-b0.1 \\
    --dpo-summary results/metrics/head_to_head/.../summary_sft_vs_dpo-b0.1.json \\
    --ppo-summary results/metrics/head_to_head/.../summary_sft_vs_ppo.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.io import DEFAULT_METRICS_DIR
from analysis.verdict import (
    arm_from_head_to_head_summary,
    build_dpo_ppo_verdict,
    write_dpo_ppo_verdict,
)
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="write dpo-vs-ppo verdict")
    p.add_argument(
        "--dpo-summary",
        type=Path,
        required=True,
        help="head-to-head summary json (sft vs dpo)",
    )
    p.add_argument(
        "--ppo-summary",
        type=Path,
        required=True,
        help="head-to-head summary json (sft vs ppo)",
    )
    p.add_argument(
        "--dpo-name",
        type=str,
        default="dpo",
        help="label for dpo arm (e.g. dpo-b0.1)",
    )
    p.add_argument("--ppo-name", type=str, default="ppo")
    p.add_argument("--metrics-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        dpo = arm_from_head_to_head_summary(args.dpo_name, args.dpo_summary)
        ppo = arm_from_head_to_head_summary(args.ppo_name, args.ppo_summary)
        verdict = build_dpo_ppo_verdict(dpo, ppo)
        write_dpo_ppo_verdict(
            verdict,
            metrics_dir / "dpo_vs_ppo_verdict.json",
            markdown_path=metrics_dir / "dpo_vs_ppo_verdict.md",
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

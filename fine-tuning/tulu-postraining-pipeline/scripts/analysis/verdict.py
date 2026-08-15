#!/usr/bin/env python3
"""decide a judged comparison from head-to-head summaries.

two shapes, matching how the arms were judged:

  DIRECT — rs-sft was judged head-to-head against dpo, so one summary answers it:
    python scripts/analysis/verdict.py \
      --arm results/metrics/head_to_head/summary_dpo_vs_rs.json \
      --out-name rs_vs_dpo

  INDIRECT — dpo and ppo were each judged against sft, never against each other:
    python scripts/analysis/verdict.py \
      --challenger results/metrics/head_to_head/summary_sft_vs_ppo.json \
      --baseline   results/metrics/head_to_head/summary_sft_vs_dpo.json \
      --out-name   dpo_vs_ppo

in every summary, model b is the arm and model a is the opponent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    DEFAULT_METRICS_DIR,
    assess_head_to_head,
    compare_arms,
    load_win_rate,
    write_verdict,
)
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="decide a judged comparison")
    p.add_argument("--arm", type=Path, default=None, help="direct head-to-head summary")
    p.add_argument("--challenger", type=Path, default=None, help="summary for the arm on trial")
    p.add_argument("--baseline", type=Path, default=None, help="summary for the arm it must beat")
    p.add_argument("--question", default=None, help="overrides the generated title")
    p.add_argument("--out-name", default="verdict")
    p.add_argument("--metrics-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.arm is None and not (args.challenger and args.baseline):
        print(
            "error: pass --arm for a direct head-to-head, "
            "or --challenger and --baseline for two arms judged against the same opponent",
            file=sys.stderr,
        )
        return 1

    metrics = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics.mkdir(parents=True, exist_ok=True)
    try:
        if args.arm is not None:
            verdict = assess_head_to_head(load_win_rate(args.arm), question=args.question)
        else:
            verdict = compare_arms(
                load_win_rate(args.challenger),
                load_win_rate(args.baseline),
                question=args.question,
            )
        write_verdict(
            verdict,
            metrics / f"{args.out_name}.json",
            markdown_path=metrics / f"{args.out_name}.md",
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

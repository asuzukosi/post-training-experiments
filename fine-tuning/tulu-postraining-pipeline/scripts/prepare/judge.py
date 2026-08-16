#!/usr/bin/env python3
"""freeze the judging prompt set every head-to-head runs on.

test_prefs minus ppo's training prompts, written as `{id, prompt}` jsonl for
`scripts/eval/head_to_head.py --prompts`. run ppo prep first, or the pool is rebuilt
in-memory to work out which ids to hold back.

examples:
  python scripts/prepare/judge.py
  python scripts/prepare/judge.py --limit 200 --output data/processed/judge_smoke.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prepare.config import load_config
from prepare.judge import DEFAULT_JUDGE_PROMPTS_PATH, prepare_judge_prompts
from prepare.paths import CONFIG_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="freeze the judging prompt set")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory with ppo.yaml (source split, seed, and the pool to exclude)",
    )
    p.add_argument("--output", type=Path, default=Path(DEFAULT_JUDGE_PROMPTS_PATH))
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="take only the first N; default is everything ppo leaves",
    )
    p.add_argument(
        "--skip-decontam",
        action="store_true",
        help="skip eval 8-gram decontam (debug only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_judge_prompts(
            load_config("ppo", args.config_dir),
            output_path=args.output,
            num_prompts=args.limit,
            skip_decontam=args.skip_decontam,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("prepare judge prompts done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

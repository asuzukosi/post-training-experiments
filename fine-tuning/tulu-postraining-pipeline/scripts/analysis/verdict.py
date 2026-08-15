#!/usr/bin/env python3
"""write a judged-win-rate verdict from head-to-head summaries.

two shapes:

  dpo vs ppo — both arms judged against SFT, compared on the difference:
    python scripts/analysis/verdict.py \
      --a dpo=results/.../summary_sft_vs_dpo.json \
      --b ppo=results/.../summary_sft_vs_ppo.json

  rs-sft vs one opponent — judged directly, so chance is 0.5:
    python scripts/analysis/verdict.py \
      --a rs_sft=results/.../summary_dpo_vs_rs.json --against-chance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    DEFAULT_METRICS_DIR,
    compare,
    compare_to_chance,
    load_win_rate,
    write_verdict,
)
from prepare.paths import resolve_path


def _arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"expected NAME=PATH, got {spec!r}")
    return name.strip(), Path(path.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="write a judged win-rate verdict")
    p.add_argument("--a", required=True, help="NAME=head-to-head summary json")
    p.add_argument("--b", default=None, help="NAME=head-to-head summary json")
    p.add_argument(
        "--against-chance",
        action="store_true",
        help="single arm judged directly against its opponent (chance 0.5)",
    )
    p.add_argument("--question", default=None)
    p.add_argument("--out-name", default="verdict")
    p.add_argument("--metrics-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics.mkdir(parents=True, exist_ok=True)
    try:
        name_a, path_a = _arm(args.a)
        arm_a = load_win_rate(name_a, path_a)
        if args.against_chance:
            question = args.question or f"does {name_a} beat its opponent?"
            verdict = compare_to_chance(arm_a, question=question)
        else:
            if args.b is None:
                print("error: pass --b, or --against-chance", file=sys.stderr)
                return 1
            name_b, path_b = _arm(args.b)
            arm_b = load_win_rate(name_b, path_b)
            question = args.question or f"{name_a} vs {name_b}"
            verdict = compare(arm_a, arm_b, question=question)
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

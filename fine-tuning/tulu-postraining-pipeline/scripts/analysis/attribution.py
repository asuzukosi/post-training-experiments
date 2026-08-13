#!/usr/bin/env python3
"""build stage-attribution table from skills (+ optional style) maps.

examples:
  python scripts/analysis/attribution.py \\
    --skills-json results/metrics/skills_map.json \\
    --style-json results/metrics/style_vs_sft_map.json

  python scripts/analysis/attribution.py \\
    --skills base=results/metrics/skills_base.json \\
    --skills sft=results/metrics/skills_sft.json \\
    --allow-incomplete
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.attribution import (
    build_stage_attribution_table,
    write_stage_attribution_table,
)
from analysis.io import DEFAULT_METRICS_DIR, merge_stage_map
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build stage-attribution table")
    p.add_argument(
        "--skills-json",
        type=Path,
        default=None,
        help='json map {"base": "skills.json", "sft": "...", ...}',
    )
    p.add_argument(
        "--skills",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="repeatable stage=skills.json override/addition",
    )
    p.add_argument(
        "--style-json",
        type=Path,
        default=None,
        help='json map {"dpo-b0.05": "style.json", ...}',
    )
    p.add_argument(
        "--style",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="repeatable stage=style.json",
    )
    p.add_argument("--metrics-dir", type=Path, default=None)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write table even if stages are missing",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skills = merge_stage_map(args.skills_json, args.skills)
    if not skills:
        print(
            "error: needs --skills-json and/or --skills STAGE=PATH",
            file=sys.stderr,
        )
        return 2
    style = merge_stage_map(args.style_json, args.style)
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        table = build_stage_attribution_table(
            skills=skills,
            style_vs_sft=style or None,
            require_complete=not args.allow_incomplete,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = write_stage_attribution_table(table, metrics_dir / "stage_attribution.json")
    print(f"attribution complete={table.complete} missing={table.missing} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

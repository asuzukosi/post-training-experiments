#!/usr/bin/env python3
"""plot raw vs length-controlled chattiness from vs-sft style reports.

examples:
  python scripts/analysis/chattiness.py \\
    --style dpo-b0.05=results/metrics/style_dpo_b005.json \\
    --style dpo-b0.1=results/metrics/style_dpo_b01.json \\
    --style ppo=results/metrics/style_ppo.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.chattiness import (
    chattiness_point_from_style_report,
    plot_length_markdown,
    plot_raw_vs_length_controlled,
    summarize_chattiness,
)
from analysis.io import DEFAULT_METRICS_DIR, DEFAULT_PLOTS_DIR, merge_stage_map, write_json
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="plot chattiness from vs-sft style reports")
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
    p.add_argument("--plots-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    style = merge_stage_map(args.style_json, args.style)
    if not style:
        print(
            "error: needs --style-json and/or --style STAGE=PATH",
            file=sys.stderr,
        )
        return 2
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    plots_dir = resolve_path(args.plots_dir) if args.plots_dir else DEFAULT_PLOTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        points = [
            chattiness_point_from_style_report(stage, path) for stage, path in style.items()
        ]
        summary_path = write_json(
            metrics_dir / "chattiness_summary.json",
            {"stages": summarize_chattiness(points)},
        )
        plot_raw_vs_length_controlled(points, plots_dir / "chattiness_raw_vs_lc.png")
        plot_length_markdown(points, plots_dir / "chattiness_length_markdown.png")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: missing dependency for analysis plots: {exc}", file=sys.stderr)
        return 1
    print(f"chattiness summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

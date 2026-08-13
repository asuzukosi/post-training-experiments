#!/usr/bin/env python3
"""kl-frontier from a bon sweep.json: gold-vs-kl, inverted-u, reusable bound.

examples:
  python scripts/analysis/kl_frontier.py \\
    --sweep results/metrics/bon_sweep/sweep.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.io import DEFAULT_METRICS_DIR, DEFAULT_PLOTS_DIR
from analysis.kl_frontier import (
    build_kl_frontier,
    plot_inverted_u,
    points_from_sweep,
    write_kl_frontier,
)
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kl-frontier from bon sweep json")
    p.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="bon sweep.json from scripts/eval/bon_sweep.py",
    )
    p.add_argument("--metrics-dir", type=Path, default=None)
    p.add_argument("--plots-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    plots_dir = resolve_path(args.plots_dir) if args.plots_dir else DEFAULT_PLOTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        frontier = build_kl_frontier(points_from_sweep(args.sweep))
        plot_path = None if args.no_plot else plots_dir / "kl_frontier_gold_vs_kl.png"
        write_kl_frontier(
            frontier,
            metrics_dir / "kl_frontier.json",
            plot_path=plot_path,
        )
        if not args.no_plot:
            plot_inverted_u(frontier, plots_dir / "inverted_u_proxy_gold.png")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

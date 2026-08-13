#!/usr/bin/env python3
"""plot + detect preference displacement.

examples:
  python scripts/analysis/displacement.py \\
    --displacement-json results/metrics/displacement_series.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.beta_plots import (
    detect_displacement,
    displacement_series_from_json,
    plot_displacement,
    plot_displacement_arms,
)
from analysis.io import DEFAULT_METRICS_DIR, DEFAULT_PLOTS_DIR, write_json
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="plot preference displacement")
    p.add_argument(
        "--displacement-json",
        type=Path,
        required=True,
        help="json list of {beta, steps, chosen_logps, rejected_logps}",
    )
    p.add_argument("--metrics-dir", type=Path, default=None)
    p.add_argument("--plots-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    plots_dir = resolve_path(args.plots_dir) if args.plots_dir else DEFAULT_PLOTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        series_list = displacement_series_from_json(args.displacement_json)
        flags = [detect_displacement(s) for s in series_list]
        flags_path = write_json(metrics_dir / "displacement_flags.json", {"arms": flags})
        print(f"displacement flags -> {flags_path}")
        for series in series_list:
            plot_displacement(series, plots_dir / f"displacement_b{series.beta:g}.png")
        if len(series_list) > 1:
            plot_displacement_arms(series_list, plots_dir / "displacement_by_beta.png")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: missing dependency for analysis plots: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

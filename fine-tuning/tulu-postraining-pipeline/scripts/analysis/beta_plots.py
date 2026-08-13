#!/usr/bin/env python3
"""plot beta vs kl/win-rate.

examples:
  python scripts/analysis/beta_plots.py \\
    --beta-arms-json results/metrics/beta_arms.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.beta_plots import beta_arms_from_json, plot_beta_vs_kl_winrate
from analysis.io import DEFAULT_PLOTS_DIR
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="plot dpo beta vs kl/win-rate")
    p.add_argument(
        "--beta-arms-json",
        type=Path,
        required=True,
        help="json list of {beta, kl?, win_rate_vs_sft?, win_rate_vs_sft_lc?}",
    )
    p.add_argument("--plots-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plots_dir = resolve_path(args.plots_dir) if args.plots_dir else DEFAULT_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        arms = beta_arms_from_json(args.beta_arms_json)
        out = plot_beta_vs_kl_winrate(arms, plots_dir / "beta_vs_kl_winrate.png")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: missing dependency for analysis plots: {exc}", file=sys.stderr)
        return 1
    print(f"beta plots -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

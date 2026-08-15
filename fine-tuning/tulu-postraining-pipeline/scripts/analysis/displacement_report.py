#!/usr/bin/env python3
"""report preference displacement for each dpo beta arm.

  python scripts/analysis/displacement_report.py --series results/metrics/displacement.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis import DEFAULT_METRICS_DIR, detect_displacement, displacement_series_from_json
from analysis.io import write_json
from prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="report dpo preference displacement")
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--metrics-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    metrics.mkdir(parents=True, exist_ok=True)
    try:
        rows = [detect_displacement(s) for s in displacement_series_from_json(args.series)]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for row in rows:
        flag = "DISPLACEMENT" if row["displacement"] else "ok"
        print(
            f"beta={row['beta']:g} chosen={row['chosen_delta']:+.4f} "
            f"rejected={row['rejected_delta']:+.4f} -> {flag}"
        )
    write_json(metrics / "displacement.json", {"arms": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

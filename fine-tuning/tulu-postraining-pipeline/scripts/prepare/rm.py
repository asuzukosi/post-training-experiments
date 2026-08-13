#!/usr/bin/env python3
"""prepare the ultrafeedback rm 20k pair subset.

examples:
  python scripts/prepare/rm.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prepare.config import load_config
from prepare.paths import CONFIG_DIR
from prepare.rm import prepare_rm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prepare ultrafeedback rm subset")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory with rm.yaml",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_rm(load_config("rm", args.config_dir))
    print("prepare rm done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

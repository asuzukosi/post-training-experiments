#!/usr/bin/env python3
"""prepare the ultrafeedback ppo 1.5k prompt pool.

examples:
  python scripts/prepare/ppo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prepare.config import load_config
from prepare.paths import CONFIG_DIR
from prepare.ppo import prepare_ppo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prepare ultrafeedback ppo prompts")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory with ppo.yaml",
    )
    p.add_argument(
        "--skip-decontam",
        action="store_true",
        help="skip eval 8-gram decontam (debug only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_ppo(load_config("ppo", args.config_dir), skip_decontam=args.skip_decontam)
    print("prepare ppo done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""prepare structured preference pairs (constraint with vs without).

examples:
  python scripts/prepare/structured.py --generator-model <sft-or-teacher>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prepare.config import load_config
from prepare.paths import CONFIG_DIR
from prepare.structured import prepare_structured


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prepare structured constraint pairs")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory with structured.yaml",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="authored jsonl override ({id, instruction, constraint})",
    )
    p.add_argument(
        "--generator-model",
        default=None,
        help="sft checkpoint or teacher used to generate both arms",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_structured(
        load_config("structured", args.config_dir),
        prompts_path=args.prompts,
        generator_model=args.generator_model,
    )
    print("prepare structured done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

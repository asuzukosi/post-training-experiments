#!/usr/bin/env python3
"""prepare sft / rm / dpo / ppo subsets and write to data/processed/.

examples:
  python scripts/prepare_data.py --sft
  python scripts/prepare_data.py --rm --dpo
  python scripts/prepare_data.py --ppo-prompts
  python scripts/prepare_data.py --all
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.prepare.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

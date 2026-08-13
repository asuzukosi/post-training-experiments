#!/usr/bin/env python3
"""best-of-n tournament to top-1; write an rs-sft dataset.

examples:
  python scripts/eval/bon.py \\
    --generations results/metrics/rs/gens.jsonl \\
    --processed-path data/processed/rs_sft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.bon import DEFAULT_BON_JUDGE_MODEL, run_bon_selection
from prepare.paths import resolve_path

DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bon tournament to top-1 rs-sft")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="jsonl of {prompt_id,prompt,completion} samples",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--processed-path", type=Path, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    out_dir = metrics_dir / "bon"
    processed = resolve_path(
        args.processed_path
        if args.processed_path is not None
        else cfg.get("bon_processed_path", "data/processed/rs_sft")
    )
    judge_model = (
        args.judge_model
        or cfg.get("bon_judge_model")
        or DEFAULT_BON_JUDGE_MODEL
    )
    run_bon_selection(
        generations_path=args.generations,
        output_dir=out_dir,
        processed_path=processed,
        judge_model=str(judge_model),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""best-of-n sweep: proxy select at 8 n-values, gold-score vs n=1.

examples:
  python scripts/eval/bon_sweep.py \\
    --generations results/metrics/bon/gens.jsonl \\
    --rm-checkpoint results/checkpoints/rm_1.5B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.bon import DEFAULT_N_VALUES, run_bon_sweep, score_proxy_incremental
from eval.judge import DEFAULT_JUDGE_MODEL
from eval.style import DEFAULT_MAX_REL_LENGTH_DIFF
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
    p = argparse.ArgumentParser(description="bon sweep: proxy + gold + kl per n")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="jsonl of {prompt_id,prompt,completion,sample_idx} samples",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--rm-checkpoint",
        type=Path,
        default=None,
        help="score missing proxy_score then sweep",
    )
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument(
        "--n-values",
        type=str,
        default=None,
        help="comma-separated n, default from eval.yaml",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    out_dir = metrics_dir / "bon_sweep"
    gens_path = resolve_path(args.generations)
    if args.rm_checkpoint is not None:
        gens_path = score_proxy_incremental(
            gens_path,
            rm_checkpoint=args.rm_checkpoint,
            output_path=out_dir / "scored.jsonl",
        )
    raw_ns = args.n_values if args.n_values is not None else cfg.get(
        "bon_n_values", DEFAULT_N_VALUES
    )
    judge_model = args.judge_model or cfg.get("judge_model") or DEFAULT_JUDGE_MODEL
    run_bon_sweep(
        generations_path=gens_path,
        output_dir=out_dir,
        n_values=raw_ns,
        judge_model=str(judge_model),
        max_rel_length_diff=float(
            cfg.get("max_rel_length_diff", DEFAULT_MAX_REL_LENGTH_DIFF)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

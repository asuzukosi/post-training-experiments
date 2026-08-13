#!/usr/bin/env python3
"""sycophancy vectors + flip-rate. extract needs a local hf checkpoint.

examples:
  python scripts/eval/steer.py extract \\
    --model results/checkpoints/<ppo> \\
    --pairs data/steer/trait_pairs.jsonl

  python scripts/eval/steer.py flip-rate \\
    --completions results/metrics/steer/flip_trials.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.io import load_jsonl
from eval.steer import (
    extract_sycophancy_vectors,
    load_trait_pairs,
    save_vectors,
    score_flip_rate,
)
from prepare.paths import resolve_path

DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sycophancy vector extract / flip-rate")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="extract v_ℓ from trait pairs")
    ex.add_argument("--model", type=str, required=True)
    ex.add_argument("--pairs", type=Path, required=True)
    ex.add_argument("--output", type=Path, default=None)
    ex.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    fr = sub.add_parser("flip-rate", help="score flip-rate from trial jsonl")
    fr.add_argument(
        "--completions",
        type=Path,
        required=True,
        help="jsonl of {id,answer,first,second} or {id,answer,trials}",
    )
    fr.add_argument("--output", type=Path, default=None)
    fr.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args(argv)


def _out_path(cfg: dict, output: Path | None, name: str) -> Path:
    if output is not None:
        return resolve_path(output)
    return resolve_path(cfg.get("metrics_dir", "results/metrics")) / "steer" / name


def cmd_extract(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_yaml(args.config)
    pairs = load_trait_pairs(args.pairs)
    print(f"loading model for sycophancy vectors: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto"
    )
    model.eval()
    vectors = extract_sycophancy_vectors(
        pairs, model=model, tokenizer=tok, model_id=str(args.model)
    )
    save_vectors(vectors, _out_path(cfg, args.output, "sycophancy_vectors.json"))
    return 0


def cmd_flip_rate(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    records = load_jsonl(args.completions)
    if not records:
        raise ValueError(f"no completions in {args.completions}")
    summary = score_flip_rate(records)
    out = _out_path(cfg, args.output, "flip_rate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)
        f.write("\n")
    print(
        f"flip_rate={summary.flip_rate} "
        f"mean_probe={summary.mean_probe_rate} "
        f"std={summary.std_probe_rate} wrote={out}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.cmd == "extract":
            return cmd_extract(args)
        return cmd_flip_rate(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""judge-reliability metrics from a judgearena jsonl.

examples:
  python scripts/eval/judge_bias.py \\
    --judge-jsonl results/metrics/head_to_head/.../judge_....jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.judge_bias import report_judge_bias_from_jsonl
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
    p = argparse.ArgumentParser(description="judge-bias report from judge jsonl")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--judge-jsonl",
        type=Path,
        required=True,
        help="existing judgearena jsonl",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    metrics_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else cfg.get("metrics_dir", "results/metrics")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = report_judge_bias_from_jsonl(args.judge_jsonl)
    out = metrics_dir / f"judge_bias_{Path(args.judge_jsonl).stem}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
        f.write("\n")
    pos = report.position.disagreement_rate
    slope = report.length.slope
    self_pref = report.self_preference.self_pref_rate
    logprob = report.logprob.agreement_rate
    print(
        "judge_bias: "
        f"position_disagreement={pos} "
        f"length_slope={slope} "
        f"self_pref={self_pref} "
        f"logprob_agree={logprob} "
        f"wrote={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

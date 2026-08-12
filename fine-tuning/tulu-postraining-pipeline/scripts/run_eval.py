#!/usr/bin/env python3
"""run eval suites (skills, reward-bench, head-to-head, style).

examples:
  # skills baseline (ifeval + mmlu)
  python scripts/run_eval.py --ifeval --mmlu --model Qwen/Qwen2.5-1.5B

  # mmlu only with drop flag vs prior baseline
  python scripts/run_eval.py --mmlu --model results/checkpoints/<sft> \\
    --baseline-mmlu-acc 0.37

  # rewardbench-chat gate
  python scripts/run_eval.py --reward-bench --rm-checkpoint results/checkpoints/<rm>

  # head-to-head (prompts jsonl: {id, prompt})
  python scripts/run_eval.py --head-to-head \\
    --a results/checkpoints/<sft> \\
    --b results/checkpoints/<dpo> \\
    --prompts data/processed/eval_prompts.jsonl \\
    --runs 3

  # style / length-control report from an existing judge jsonl
  python scripts/run_eval.py --style --judge-jsonl results/metrics/head_to_head/.../judge_....jsonl \\
    --length-control
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

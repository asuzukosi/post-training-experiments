#!/usr/bin/env python3
"""run analysis reports (attribution, beta/displacement, chattiness, verdict).

examples:
  # stage attribution (skills map required; style map for preference stages)
  python scripts/run_analysis.py --attribution \\
    --skills-json results/metrics/skills_map.json \\
    --style-json results/metrics/style_vs_sft_map.json

  # same with repeatable overrides
  python scripts/run_analysis.py --attribution \\
    --skills base=results/metrics/skills_base.json \\
    --skills sft=results/metrics/skills_sft.json \\
    --allow-incomplete

  # beta vs kl/win-rate
  python scripts/run_analysis.py --beta-plots \\
    --beta-arms-json results/metrics/beta_arms.json

  # preference displacement curves + flags
  python scripts/run_analysis.py --displacement \\
    --displacement-json results/metrics/displacement_series.json

  # chattiness from vs-sft style reports
  python scripts/run_analysis.py --chattiness \\
    --style dpo-b0.05=results/metrics/style_dpo_b005.json \\
    --style dpo-b0.1=results/metrics/style_dpo_b01.json \\
    --style ppo=results/metrics/style_ppo.json

  # dpo-vs-ppo equal-data verdict (pick one dpo beta arm)
  python scripts/run_analysis.py --verdict \\
    --dpo-name dpo-b0.1 \\
    --dpo-summary results/metrics/head_to_head/.../summary_sft_vs_dpo-b0.1.json \\
    --ppo-summary results/metrics/head_to_head/.../summary_sft_vs_ppo.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.analysis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

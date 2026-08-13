# tulu post-training pipeline (sft → rm → dpo×β → ppo)

reproducible 1.5b recipe. success = released checkpoints + stage attribution + dpo-vs-ppo verdict (not beating qwen instruct).

## analysis entry

```bash
python scripts/analysis/attribution.py \
  --skills-json results/metrics/skills_map.json \
  --style-json results/metrics/style_vs_sft_map.json

python scripts/analysis/beta_plots.py --beta-arms-json results/metrics/beta_arms.json
python scripts/analysis/displacement.py --displacement-json results/metrics/displacement_series.json
python scripts/analysis/chattiness.py --style-json results/metrics/style_vs_sft_map.json
python scripts/analysis/verdict.py \
  --dpo-name dpo-b0.1 \
  --dpo-summary results/metrics/head_to_head/.../summary_sft_vs_dpo.json \
  --ppo-summary results/metrics/head_to_head/.../summary_sft_vs_ppo.json
```

## findings (fill after phase 8)

placeholder — paste from `results/metrics/` + `results/plots/` after full evals:

- stage attribution (where format / skills / style emerge)
- β sensitivity + preference displacement
- chattiness (raw vs length-controlled win-rate)
- dpo-vs-ppo equal-data verdict (primary = length-controlled, with cis)

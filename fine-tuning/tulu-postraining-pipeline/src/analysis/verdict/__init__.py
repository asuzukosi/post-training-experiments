"""verdict writers: shared stats/arms, then dpo-vs-ppo and rs-vs-dpo."""
from analysis.verdict.arms import ArmVsSft, arm_from_head_to_head_summary
from analysis.verdict.dpo_ppo import (
    DpoPpoVerdict,
    MetricVerdict,
    build_dpo_ppo_verdict,
    compare_metric,
    write_dpo_ppo_verdict,
)
from analysis.verdict.rs_dpo import (
    RS_DPO_CLAIM,
    RsDpoMetric,
    RsDpoVerdict,
    build_rs_dpo_verdict,
    compare_rs_against_dpo,
    winner_against_chance,
    write_rs_dpo_verdict,
)
from analysis.verdict.stats import RunSummary, summarize_runs

__all__ = [
    "ArmVsSft",
    "DpoPpoVerdict",
    "MetricVerdict",
    "RS_DPO_CLAIM",
    "RsDpoMetric",
    "RsDpoVerdict",
    "RunSummary",
    "arm_from_head_to_head_summary",
    "build_dpo_ppo_verdict",
    "build_rs_dpo_verdict",
    "compare_metric",
    "compare_rs_against_dpo",
    "summarize_runs",
    "winner_against_chance",
    "write_dpo_ppo_verdict",
    "write_rs_dpo_verdict",
]

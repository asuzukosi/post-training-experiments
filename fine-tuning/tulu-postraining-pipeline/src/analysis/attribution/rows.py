"""one row of the attribution table, built from a stage's evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from analysis.attribution.inputs import StageEvaluation
from analysis.attribution.stages import PREFERENCE_STAGES

@dataclass
class AttributionRow:
    """one generative stage's format / skills / style metrics."""

    stage: str # base, sft, dpo-b0.05, dpo-b0.1, ppo
    ifeval: float | None = None # ifeval prompt_level_strict_acc
    mmlu: float | None = None # mmlu_acc
    mean_chars: float | None = None # style_b.mean_chars
    markdown_rate: float | None = None # style_b.markdown_rate
    sft_win_rate: float | None = None # raw.win_rate_b
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_row(
    evaluation: StageEvaluation,
    *,
    missing: list[str],
) -> AttributionRow:
    stage = evaluation.stage
    if not evaluation.has_benchmarks:
        missing.append(f"{stage}:benchmarks")
        return AttributionRow(stage=stage, notes="missing benchmark scores")

    ifeval, mmlu = evaluation.ifeval_acc, evaluation.mmlu_acc
    mean_chars = md_rate = win_raw = None
    notes = ""

    judged = evaluation.sft_comparison
    if judged is not None:
        mean_chars, md_rate, win_raw = judged.mean_chars, judged.markdown_rate, judged.win_rate
    elif stage in PREFERENCE_STAGES:
        missing.append(f"{stage}:sft_comparison")
        notes = "missing preference comparison vs sft"

    if ifeval is None:
        missing.append(f"{stage}:ifeval")
    if mmlu is None:
        missing.append(f"{stage}:mmlu")

    return AttributionRow(
        stage=stage,
        ifeval=ifeval,
        mmlu=mmlu,
        mean_chars=mean_chars,
        markdown_rate=md_rate,
        sft_win_rate=win_raw,
        notes=notes,
    )

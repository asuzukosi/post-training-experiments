"""how each stage's metrics moved against the stage before it."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from analysis.attribution.rows import AttributionRow
from analysis.attribution.stages import PREFERENCE_STAGES

@dataclass
class StageDelta:
    """how one stage's metrics moved against the stage it is compared with.

    every field here is a SIGNED CHANGE except `judged_win_rate`: positive means the
    metric went up between `versus` and `stage`.

    `judged_win_rate` is an absolute rate, not a change. a win-rate only exists
    pairwise, so there is no earlier win-rate to subtract from — it is `stage`'s share
    of decisive judged pairs against `versus`, where 0.5 is parity.
    """

    stage: str
    versus: str
    ifeval: float | None = None
    mmlu: float | None = None
    mean_chars: float | None = None
    markdown_rate: float | None = None
    judged_win_rate: float | None = None

    @property
    def label(self) -> str:
        return f"{self.stage}_vs_{self.versus}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "versus": self.versus,
            "changes": {
                "ifeval": self.ifeval,
                "mmlu": self.mmlu,
                "mean_chars": self.mean_chars,
                "markdown_rate": self.markdown_rate,
            },
            "judged_win_rate": self.judged_win_rate,
        }


def calculate_diff(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None:
        return None
    return curr - prev


def build_attribution_delta(before: AttributionRow, after: AttributionRow) -> StageDelta:
    """change from `before` to `after`, field by field: `after` minus `before`.

    the judged win-rate is carried through from `after` rather than subtracted — it is
    already a comparison, so there is nothing on `before` to subtract from. it comes out
    None whenever `after` has no judged report, which is the case for sft itself.
    """
    return StageDelta(
        stage=after.stage,
        versus=before.stage,
        ifeval=calculate_diff(after.ifeval, before.ifeval),
        mmlu=calculate_diff(after.mmlu, before.mmlu),
        mean_chars=calculate_diff(after.mean_chars, before.mean_chars),
        markdown_rate=calculate_diff(after.markdown_rate, before.markdown_rate),
        judged_win_rate=after.sft_win_rate,
    )


def compute_stage_deltas(rows: Sequence[AttributionRow]) -> list[StageDelta]:
    """what sft added over base, and what each preference stage added over sft.

    preference stages are measured against SFT rather than base: measuring them against
    base would credit them with everything sft already did.
    """
    by_stage: dict[str, AttributionRow] = {r.stage: r for r in rows} # stage -> row
    deltas: list[StageDelta] = []

    base, sft = by_stage.get("base"), by_stage.get("sft")
    if base is not None and sft is not None:
        deltas.append(build_attribution_delta(base, sft))

    if sft is None:
        return deltas

    for stage in PREFERENCE_STAGES:
        row = by_stage.get(stage)
        if row is None:
            continue
        deltas.append(build_attribution_delta(sft, row))
    return deltas

"""assemble the rows into the deliverable, and gate on completeness."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.attribution.deltas import StageDelta, compute_stage_deltas
from analysis.attribution.inputs import StageEvaluation
from analysis.attribution.rows import AttributionRow, build_row
from analysis.attribution.stages import GENERATIVE_STAGES, PREFERENCE_STAGES
from analysis.io import DEFAULT_METRICS_DIR, write_json
from prepare.paths import resolve_path

DEFAULT_ATTRIBUTION_PATH = DEFAULT_METRICS_DIR / "stage_attribution.json" # where the attribution table is written to

# a json file, or its already-loaded contents. only the loaders below accept this;
# everything past them holds parsed values.


@dataclass
class AttributionTable:
    """stage-attribution deliverable + completeness gate."""

    rows: list[AttributionRow]
    complete: bool
    missing: list[str] = field(default_factory=list)
    deltas: list[StageDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "missing": list(self.missing),
            "rows": [r.to_dict() for r in self.rows],
            "deltas": [d.to_dict() for d in self.deltas],
        }

    def raise_if_incomplete(self) -> None:
        if self.complete:
            return
        raise RuntimeError(
            "stage attribution incomplete; missing metrics for: "
            + ", ".join(self.missing)
        )


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _blocking_missing(
    missing: Sequence[str],
    required_stages: Sequence[str],
    *,
    require_judged: bool,
) -> list[str]:
    """benchmark holes always block; a missing judged report blocks when required."""
    blockers: list[str] = []
    for item in missing:
        stage = item.split(":", 1)[0]
        if stage not in required_stages:
            continue
        if item.endswith((":benchmarks", ":ifeval", ":mmlu")):
            blockers.append(item)
        elif require_judged and item.endswith(":sft_comparison") and stage in PREFERENCE_STAGES:
            blockers.append(item)
    return _unique(blockers)


def build_stage_attribution_table(
    evaluations: Sequence[StageEvaluation],
    *,
    required_stages: Sequence[str] = GENERATIVE_STAGES,
    require_complete: bool = True,
) -> AttributionTable:
    """build one row per generative stage from its evaluation artifacts.

    reward-model quality is not in this table — it is gated separately on
    rewardbench-chat, because the rm is not a generative stage.
    """
    by_stage = {e.stage: e for e in evaluations}
    ordered = list(required_stages) + [
        s for s in by_stage if s not in required_stages
    ]

    missing: list[str] = []
    rows = [
        build_row(
            by_stage.get(stage, StageEvaluation(stage=stage)),
            missing=missing,
        )
        for stage in ordered
    ]

    missing = _unique(missing)
    blockers = _blocking_missing(
        missing,
        required_stages,
        require_judged=require_complete,
    )
    table = AttributionTable(
        rows=rows,
        complete=len(blockers) == 0,
        missing=missing,
        deltas=compute_stage_deltas(rows),
    )
    if require_complete:
        table.raise_if_incomplete()
    return table


def write_stage_attribution_table(
    table: AttributionTable,
    path: str | Path | None = None,
) -> Path:
    """write attribution json under results/metrics/."""
    out = DEFAULT_ATTRIBUTION_PATH if path is None else resolve_path(path)
    write_json(out, table.to_dict())
    print(f"wrote stage attribution -> {out} complete={table.complete}")
    return out

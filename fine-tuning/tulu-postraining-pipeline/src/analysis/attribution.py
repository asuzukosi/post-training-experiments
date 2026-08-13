"""stage-attribution table: format / skills / style for generative stages only.

rm is a scalar reward model — evaluated only via the rewardbench-chat gate,
not as a row in this table.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analysis.io import (
    DEFAULT_METRICS_DIR,
    as_float,
    load_json_mapping,
    write_json,
)
from prepare.paths import resolve_path

GENERATIVE_STAGES = (
    "base",
    "sft",
    "dpo-b0.05",
    "dpo-b0.1",
    "ppo",
)
PREFERENCE_STAGES = ("dpo-b0.05", "dpo-b0.1", "ppo")
DEFAULT_ATTRIBUTION_PATH = DEFAULT_METRICS_DIR / "stage_attribution.json"

Artifact = str | Path | Mapping[str, Any]


@dataclass
class AttributionRow:
    """one generative stage's format / skills / style metrics."""

    stage: str
    format_ifeval: float | None = None
    skills_mmlu: float | None = None
    style_mean_chars: float | None = None
    style_markdown_rate: float | None = None
    style_win_rate_vs_sft_raw: float | None = None
    style_win_rate_vs_sft_lc: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionTable:
    """stage-attribution deliverable + completeness gate."""

    rows: list[AttributionRow]
    complete: bool
    missing: list[str] = field(default_factory=list)
    deltas: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "missing": list(self.missing),
            "rows": [r.to_dict() for r in self.rows],
            "deltas": self.deltas,
        }

    def raise_if_incomplete(self) -> None:
        if self.complete:
            return
        raise RuntimeError(
            "stage attribution incomplete; missing metrics for: "
            + ", ".join(self.missing)
        )


def _parse_skills(payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    return as_float(payload.get("ifeval_prompt_strict")), as_float(payload.get("mmlu_acc"))


def _parse_style_vs_sft(
    payload: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    """style report where model b is the stage model vs sft as a."""
    style_b = payload.get("style_b") or {}
    raw = payload.get("raw") or {}
    lc = payload.get("length_controlled") or {}
    return (
        as_float(style_b.get("mean_chars")),
        as_float(style_b.get("markdown_rate")),
        as_float(raw.get("win_rate_b")),
        as_float(lc.get("win_rate_b")),
    )


def _delta(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None:
        return None
    return curr - prev


def _ordered_stages(
    required: Sequence[str],
    skills: Mapping[str, Artifact],
    style_vs_sft: Mapping[str, Artifact],
) -> list[str]:
    stages = list(required)
    for stage in list(skills) + list(style_vs_sft):
        if stage not in stages:
            stages.append(stage)
    return stages


def _row_from_skills_and_style(
    stage: str,
    *,
    skills: Mapping[str, Artifact],
    style_vs_sft: Mapping[str, Artifact],
    missing: list[str],
) -> AttributionRow:
    if stage not in skills:
        missing.append(f"{stage}:skills")
        return AttributionRow(stage=stage, notes="missing skills metrics")

    ifeval, mmlu = _parse_skills(load_json_mapping(skills[stage]))
    mean_chars = md_rate = win_raw = win_lc = None
    notes = ""

    if stage in style_vs_sft:
        mean_chars, md_rate, win_raw, win_lc = _parse_style_vs_sft(
            load_json_mapping(style_vs_sft[stage])
        )
    elif stage in PREFERENCE_STAGES:
        missing.append(f"{stage}:style_vs_sft")
        notes = "missing style_vs_sft metrics"

    if ifeval is None:
        missing.append(f"{stage}:format_ifeval")
    if mmlu is None:
        missing.append(f"{stage}:skills_mmlu")

    return AttributionRow(
        stage=stage,
        format_ifeval=ifeval,
        skills_mmlu=mmlu,
        style_mean_chars=mean_chars,
        style_markdown_rate=md_rate,
        style_win_rate_vs_sft_raw=win_raw,
        style_win_rate_vs_sft_lc=win_lc,
        notes=notes,
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
    require_style: bool,
) -> list[str]:
    """skills holes always block; preference style holes block when require_style."""
    blockers: list[str] = []
    for item in missing:
        stage = item.split(":", 1)[0]
        if stage not in required_stages:
            continue
        if item.endswith((":skills", ":format_ifeval", ":skills_mmlu")):
            blockers.append(item)
        elif require_style and item.endswith(":style_vs_sft") and stage in PREFERENCE_STAGES:
            blockers.append(item)
    return _unique(blockers)


def compute_stage_deltas(
    rows: Sequence[AttributionRow],
) -> dict[str, dict[str, float | None]]:
    """sft vs base and preference-stage vs sft deltas for the attribution report."""
    by_stage = {r.stage: r for r in rows}
    out: dict[str, dict[str, float | None]] = {}

    base, sft = by_stage.get("base"), by_stage.get("sft")
    if base is not None and sft is not None:
        out["sft_vs_base"] = {
            "format_ifeval": _delta(sft.format_ifeval, base.format_ifeval),
            "skills_mmlu": _delta(sft.skills_mmlu, base.skills_mmlu),
        }

    if sft is None:
        return out

    for stage in PREFERENCE_STAGES:
        row = by_stage.get(stage)
        if row is None:
            continue
        out[f"{stage}_vs_sft"] = {
            "format_ifeval": _delta(row.format_ifeval, sft.format_ifeval),
            "skills_mmlu": _delta(row.skills_mmlu, sft.skills_mmlu),
            "style_win_rate_vs_sft_raw": row.style_win_rate_vs_sft_raw,
            "style_win_rate_vs_sft_lc": row.style_win_rate_vs_sft_lc,
            "style_mean_chars_delta": _delta(row.style_mean_chars, sft.style_mean_chars),
            "style_markdown_rate_delta": _delta(
                row.style_markdown_rate, sft.style_markdown_rate
            ),
        }
    return out


def build_stage_attribution_table(
    *,
    skills: Mapping[str, Artifact],
    style_vs_sft: Mapping[str, Artifact] | None = None,
    rm_gate: Artifact | None = None,
    required_stages: Sequence[str] = GENERATIVE_STAGES,
    require_complete: bool = True,
) -> AttributionTable:
    """build format/skills/style rows for generative stages only.

    `skills`: stage -> skills json path/dict (`ifeval_prompt_strict`, `mmlu_acc`)
    `style_vs_sft`: stage -> style report with that stage as model b vs sft as a

    rm quality stays on the rewardbench-chat gate, not this table.
    """
    style_vs_sft = style_vs_sft or {}
    missing: list[str] = []
    rows = [
        _row_from_skills_and_style(
            stage,
            skills=skills,
            style_vs_sft=style_vs_sft,
            missing=missing,
        )
        for stage in _ordered_stages(required_stages, skills, style_vs_sft)
    ]

    missing = _unique(missing)
    blockers = _blocking_missing(
        missing,
        required_stages,
        require_style=require_complete,
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

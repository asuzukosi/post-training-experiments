"""stage attribution: what each training stage added, measured against the last.

    stages.py   which stages exist, and which are preference stages
    inputs.py   what evaluation produced for a stage, parsed off disk
    rows.py     one table row per stage
    deltas.py   the subtractions between stages
    table.py    assembly, the completeness gate, and writing the deliverable
"""
from analysis.attribution.deltas import (
    StageDelta,
    build_attribution_delta,
    calculate_diff,
    compute_stage_deltas,
)
from analysis.attribution.inputs import (
    JsonSource,
    JudgedComparison,
    StageEvaluation,
)
from analysis.attribution.rows import AttributionRow, build_row
from analysis.attribution.stages import GENERATIVE_STAGES, PREFERENCE_STAGES
from analysis.attribution.table import (
    DEFAULT_ATTRIBUTION_PATH,
    AttributionTable,
    build_stage_attribution_table,
    write_stage_attribution_table,
)

__all__ = [
    "DEFAULT_ATTRIBUTION_PATH",
    "GENERATIVE_STAGES",
    "PREFERENCE_STAGES",
    "AttributionRow",
    "AttributionTable",
    "JsonSource",
    "JudgedComparison",
    "StageDelta",
    "StageEvaluation",
    "build_attribution_delta",
    "build_row",
    "build_stage_attribution_table",
    "calculate_diff",
    "compute_stage_deltas",
    "write_stage_attribution_table",
]

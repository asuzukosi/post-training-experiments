"""dpo beta arms and the preference-displacement detector.

displacement is both chosen AND rejected log-probs falling: the policy moved away
from the whole preference pair rather than learning to prefer one side. figures are
produced ad hoc from this data; nothing here imports matplotlib.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.io import DEFAULT_PLOTS_DIR, resolve_output_path


@dataclass
class BetaArmPoint:
    """one dpo beta arm's summary metrics for the beta trade-off plot."""

    beta: float
    kl: float | None = None
    win_rate_vs_sft: float | None = None
    win_rate_vs_sft_lc: float | None = None


@dataclass
class DisplacementSeries:
    """per-step chosen/rejected log-probs for one beta arm (displacement signal)."""

    beta: float
    steps: list[int]
    chosen_logps: list[float]
    rejected_logps: list[float]

    def __post_init__(self) -> None:
        n = len(self.steps)
        if len(self.chosen_logps) != n or len(self.rejected_logps) != n:
            raise ValueError(
                "displacement series length mismatch: "
                f"steps={n} chosen={len(self.chosen_logps)} "
                f"rejected={len(self.rejected_logps)}"
            )


def detect_displacement(series: DisplacementSeries) -> dict[str, Any]:
    """end-vs-start check: both chosen and rejected logps fell => displacement."""
    if len(series.steps) < 2:
        return {
            "beta": series.beta,
            "chosen_fell": False,
            "rejected_fell": False,
            "displacement": False,
            "chosen_delta": 0.0,
            "rejected_delta": 0.0,
        }

    chosen_delta = series.chosen_logps[-1] - series.chosen_logps[0]
    rejected_delta = series.rejected_logps[-1] - series.rejected_logps[0]
    chosen_fell = chosen_delta < 0
    rejected_fell = rejected_delta < 0
    return {
        "beta": series.beta,
        "chosen_fell": chosen_fell,
        "rejected_fell": rejected_fell,
        "displacement": chosen_fell and rejected_fell,
        "chosen_delta": chosen_delta,
        "rejected_delta": rejected_delta,
    }


def beta_arms_from_json(path: str | Path) -> list[BetaArmPoint]:
    """load [{beta, kl?, win_rate_vs_sft?, win_rate_vs_sft_lc?}, ...]"""
    from analysis.io import load_json_list

    arms: list[BetaArmPoint] = []
    for row in load_json_list(path):
        if not isinstance(row, dict):
            raise ValueError(f"beta arm must be an object, got {row!r}")
        if "beta" not in row:
            raise ValueError(f"beta arm missing beta: {row!r}")
        arms.append(
            BetaArmPoint(
                beta=float(row["beta"]),
                kl=None if row.get("kl") is None else float(row["kl"]),
                win_rate_vs_sft=(
                    None
                    if row.get("win_rate_vs_sft") is None
                    else float(row["win_rate_vs_sft"])
                ),
                win_rate_vs_sft_lc=(
                    None
                    if row.get("win_rate_vs_sft_lc") is None
                    else float(row["win_rate_vs_sft_lc"])
                ),
            )
        )
    return arms


def displacement_series_from_json(path: str | Path) -> list[DisplacementSeries]:
    """load [{beta, steps, chosen_logps, rejected_logps}, ...]"""
    from analysis.io import load_json_list

    series_list: list[DisplacementSeries] = []
    for row in load_json_list(path):
        if not isinstance(row, dict):
            raise ValueError(f"displacement series must be an object, got {row!r}")
        series_list.append(
            DisplacementSeries(
                beta=float(row["beta"]),
                steps=[int(x) for x in row["steps"]],
                chosen_logps=[float(x) for x in row["chosen_logps"]],
                rejected_logps=[float(x) for x in row["rejected_logps"]],
            )
        )
    return series_list

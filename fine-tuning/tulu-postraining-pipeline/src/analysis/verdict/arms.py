"""load a preference arm from a head-to-head summary json."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.io import as_float, load_json_mapping


@dataclass
class ArmVsSft:
    """one method's win-rates across repeated judge runs."""

    name: str
    win_rates_raw: list[float]
    win_rates_lc: list[float] = field(default_factory=list)
    kl: float | None = None
    wall_clock_hours: float | None = None

    def __post_init__(self) -> None:
        if not self.win_rates_raw:
            raise ValueError(f"{self.name}: win_rates_raw must be non-empty")


def extract_win_rates(reports: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """key is 'raw' or 'length_controlled'; uses win_rate_b."""
    out: list[float] = []
    for row in reports:
        block = row.get(key) or {}
        value = as_float(block.get("win_rate_b"))
        if value is not None:
            out.append(value)
    return out


def arm_from_head_to_head_summary(
    name: str,
    summary: str | Path | Mapping[str, Any],
    *,
    kl: float | None = None,
    wall_clock_hours: float | None = None,
) -> ArmVsSft:
    """build an arm from a head-to-head summary json (model b = the named method)."""
    payload = load_json_mapping(summary)
    reports = payload.get("reports") or []
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"{name}: head-to-head summary missing reports")
    raw = extract_win_rates(reports, "raw")
    lc = extract_win_rates(reports, "length_controlled")
    if not raw:
        raise ValueError(f"{name}: no raw win_rate_b in reports")
    return ArmVsSft(
        name=name,
        win_rates_raw=raw,
        win_rates_lc=lc,
        kl=kl,
        wall_clock_hours=wall_clock_hours,
    )

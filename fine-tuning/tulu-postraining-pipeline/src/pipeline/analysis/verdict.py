"""dpo-vs-ppo equal-data verdict from vs-sft head-to-head runs (with cis)."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pipeline.analysis.io import (
    DEFAULT_METRICS_DIR,
    as_float,
    load_json_mapping,
    write_json,
)
from pipeline.prepare.paths import resolve_path

# student-t 95% two-sided critical values by sample size n (df = n-1)
_T95_BY_N = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}

VerdictLabel = Literal["dpo", "ppo", "tie"]
DEFAULT_VERDICT_PATH = DEFAULT_METRICS_DIR / "dpo_vs_ppo_verdict.json"


@dataclass
class ArmVsSft:
    """one method's vs-sft win-rates across repeated judge runs."""

    name: str
    win_rates_raw: list[float]
    win_rates_lc: list[float] = field(default_factory=list)
    kl: float | None = None
    wall_clock_hours: float | None = None

    def __post_init__(self) -> None:
        if not self.win_rates_raw:
            raise ValueError(f"{self.name}: win_rates_raw must be non-empty")


@dataclass
class RunSummary:
    """mean / std / 95% ci over repeated runs."""

    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricVerdict:
    """comparison on one metric (raw or length-controlled)."""

    metric: str
    dpo: RunSummary
    ppo: RunSummary
    delta_ppo_minus_dpo: float
    delta_ci95_low: float
    delta_ci95_high: float
    winner: VerdictLabel
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "dpo": self.dpo.to_dict(),
            "ppo": self.ppo.to_dict(),
            "delta_ppo_minus_dpo": self.delta_ppo_minus_dpo,
            "delta_ci95_low": self.delta_ci95_low,
            "delta_ci95_high": self.delta_ci95_high,
            "winner": self.winner,
            "note": self.note,
        }


@dataclass
class DpoPpoVerdict:
    """headline equal-data verdict (primary = length-controlled)."""

    dpo_name: str
    ppo_name: str
    raw: MetricVerdict
    length_controlled: MetricVerdict | None
    primary_winner: VerdictLabel
    primary_metric: str
    secondary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dpo_name": self.dpo_name,
            "ppo_name": self.ppo_name,
            "primary_metric": self.primary_metric,
            "primary_winner": self.primary_winner,
            "raw": self.raw.to_dict(),
            "length_controlled": (
                None if self.length_controlled is None else self.length_controlled.to_dict()
            ),
            "secondary": self.secondary,
        }

    def to_markdown(self) -> str:
        lines = [
            "# dpo vs ppo verdict (equal data vs sft)",
            "",
            f"- dpo arm: `{self.dpo_name}`",
            f"- ppo arm: `{self.ppo_name}`",
            f"- primary metric: `{self.primary_metric}`",
            f"- primary winner: **{self.primary_winner}**",
            "",
            "## raw win-rate vs sft",
            _metric_md(self.raw),
        ]
        if self.length_controlled is not None:
            lines.extend(
                [
                    "",
                    "## length-controlled win-rate vs sft",
                    _metric_md(self.length_controlled),
                ]
            )
        if self.secondary:
            lines.extend(["", "## secondary", f"```json\n{_pretty(self.secondary)}\n```"])
        lines.append("")
        return "\n".join(lines)


def _pretty(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(payload), indent=2)


def _metric_md(m: MetricVerdict) -> str:
    return "\n".join(
        [
            f"- dpo: mean={m.dpo.mean:.4f} ± std={m.dpo.std:.4f} "
            f"ci95=[{m.dpo.ci95_low:.4f}, {m.dpo.ci95_high:.4f}] (n={m.dpo.n})",
            f"- ppo: mean={m.ppo.mean:.4f} ± std={m.ppo.std:.4f} "
            f"ci95=[{m.ppo.ci95_low:.4f}, {m.ppo.ci95_high:.4f}] (n={m.ppo.n})",
            f"- delta (ppo - dpo): {m.delta_ppo_minus_dpo:.4f} "
            f"ci95=[{m.delta_ci95_low:.4f}, {m.delta_ci95_high:.4f}]",
            f"- winner: `{m.winner}` ({m.note})",
        ]
    )


def _t_crit_95(n: int) -> float:
    if n < 2:
        return float("inf")
    return _T95_BY_N.get(n, 1.96)


def summarize_runs(values: Sequence[float]) -> RunSummary:
    """mean/std and 95% ci of the mean over repeated runs."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("values must be non-empty")
    mean = sum(xs) / n
    if n == 1:
        return RunSummary(n=1, mean=mean, std=0.0, ci95_low=mean, ci95_high=mean)
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    half = _t_crit_95(n) * se
    return RunSummary(
        n=n,
        mean=mean,
        std=std,
        ci95_low=mean - half,
        ci95_high=mean + half,
    )


def _winner_from_delta_ci(delta: float, lo: float, hi: float) -> tuple[VerdictLabel, str]:
    """if 0 is outside the delta ci, declare a winner; else tie."""
    if lo > 0:
        return "ppo", "delta ci excludes 0 (ppo ahead)"
    if hi < 0:
        return "dpo", "delta ci excludes 0 (dpo ahead)"
    return "tie", "delta ci includes 0 (indistinguishable at 95%)"


def compare_metric(
    *,
    metric: str,
    dpo_values: Sequence[float],
    ppo_values: Sequence[float],
    dpo_summary: RunSummary | None = None,
    ppo_summary: RunSummary | None = None,
) -> MetricVerdict:
    dpo = dpo_summary or summarize_runs(dpo_values)
    ppo = ppo_summary or summarize_runs(ppo_values)
    delta = ppo.mean - dpo.mean
    # unpaired se of difference (independent runs)
    se_dpo = 0.0 if dpo.n < 2 else dpo.std / math.sqrt(dpo.n)
    se_ppo = 0.0 if ppo.n < 2 else ppo.std / math.sqrt(ppo.n)
    se_delta = math.sqrt(se_dpo**2 + se_ppo**2)
    # conservative: use smaller n for t
    n_eff = min(dpo.n, ppo.n)
    half = _t_crit_95(n_eff) * se_delta if n_eff >= 2 else 0.0
    lo, hi = delta - half, delta + half
    winner, note = _winner_from_delta_ci(delta, lo, hi)
    return MetricVerdict(
        metric=metric,
        dpo=dpo,
        ppo=ppo,
        delta_ppo_minus_dpo=delta,
        delta_ci95_low=lo,
        delta_ci95_high=hi,
        winner=winner,
        note=note,
    )


def _extract_win_rates(reports: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """key is 'raw' or 'length_controlled'; uses win_rate_b (stage vs sft)."""
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
    """build an arm from a head-to-head summary json (model b = preference stage)."""
    payload = load_json_mapping(summary)
    reports = payload.get("reports") or []
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"{name}: head-to-head summary missing reports")
    raw = _extract_win_rates(reports, "raw")
    lc = _extract_win_rates(reports, "length_controlled")
    if not raw:
        raise ValueError(f"{name}: no raw win_rate_b in reports")
    return ArmVsSft(
        name=name,
        win_rates_raw=raw,
        win_rates_lc=lc,
        kl=kl,
        wall_clock_hours=wall_clock_hours,
    )


def build_dpo_ppo_verdict(
    dpo: ArmVsSft,
    ppo: ArmVsSft,
    *,
    prefer_length_controlled: bool = True,
) -> DpoPpoVerdict:
    """compare dpo vs ppo using equal-prompt vs-sft win-rates (+ cis)."""
    raw = compare_metric(
        metric="raw_win_rate_vs_sft",
        dpo_values=dpo.win_rates_raw,
        ppo_values=ppo.win_rates_raw,
    )

    lc: MetricVerdict | None = None
    if dpo.win_rates_lc and ppo.win_rates_lc:
        lc = compare_metric(
            metric="length_controlled_win_rate_vs_sft",
            dpo_values=dpo.win_rates_lc,
            ppo_values=ppo.win_rates_lc,
        )

    if prefer_length_controlled and lc is not None:
        primary_metric = lc.metric
        primary_winner = lc.winner
    else:
        primary_metric = raw.metric
        primary_winner = raw.winner

    secondary: dict[str, Any] = {
        "dpo_kl": dpo.kl,
        "ppo_kl": ppo.kl,
        "dpo_wall_clock_hours": dpo.wall_clock_hours,
        "ppo_wall_clock_hours": ppo.wall_clock_hours,
    }
    return DpoPpoVerdict(
        dpo_name=dpo.name,
        ppo_name=ppo.name,
        raw=raw,
        length_controlled=lc,
        primary_winner=primary_winner,
        primary_metric=primary_metric,
        secondary=secondary,
    )


def write_dpo_ppo_verdict(
    verdict: DpoPpoVerdict,
    path: str | Path | None = None,
    *,
    markdown_path: str | Path | None = None,
) -> Path:
    """write verdict json (and optional markdown) under results/metrics/."""
    out = DEFAULT_VERDICT_PATH if path is None else resolve_path(path)
    write_json(out, verdict.to_dict())
    print(f"wrote dpo-vs-ppo verdict -> {out} winner={verdict.primary_winner}")

    if markdown_path is not None:
        md_out = resolve_path(markdown_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(verdict.to_markdown(), encoding="utf-8")
        print(f"wrote dpo-vs-ppo verdict markdown -> {md_out}")
    return out

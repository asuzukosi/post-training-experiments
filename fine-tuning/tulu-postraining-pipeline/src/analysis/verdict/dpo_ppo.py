"""dpo-vs-ppo equal-data verdict from vs-sft head-to-head runs (with cis)."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from analysis.io import DEFAULT_METRICS_DIR, write_json
from analysis.verdict.arms import ArmVsSft
from analysis.verdict.stats import RunSummary, summarize_runs, t_crit_95
from prepare.paths import resolve_path

VerdictLabel = Literal["dpo", "ppo", "tie"]
DEFAULT_VERDICT_PATH = DEFAULT_METRICS_DIR / "dpo_vs_ppo_verdict.json"


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
    se_dpo = 0.0 if dpo.n < 2 else dpo.std / math.sqrt(dpo.n)
    se_ppo = 0.0 if ppo.n < 2 else ppo.std / math.sqrt(ppo.n)
    se_delta = math.sqrt(se_dpo**2 + se_ppo**2)
    n_eff = min(dpo.n, ppo.n)
    half = t_crit_95(n_eff) * se_delta if n_eff >= 2 else 0.0
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

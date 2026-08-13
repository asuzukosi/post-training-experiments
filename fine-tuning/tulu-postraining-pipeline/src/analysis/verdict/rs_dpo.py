"""rs-sft vs dpo verdict from a direct head-to-head (win_rate_b = rs)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from analysis.io import DEFAULT_METRICS_DIR, write_json
from analysis.verdict.arms import ArmVsSft
from analysis.verdict.stats import RunSummary, summarize_runs
from prepare.paths import resolve_path

RsDpoWinner = Literal["rs_sft", "dpo", "tie"]
DEFAULT_RS_VERDICT_PATH = DEFAULT_METRICS_DIR / "rs_sft_vs_dpo_verdict.json"
RS_DPO_CLAIM = (
    "rs-sft from a 32b teacher vs dpo on that teacher's preferences; "
    "the teacher is not the policy, so this is not 'rejection sampling beats dpo' in general"
)


@dataclass
class RsDpoMetric:
    """rs win-rate vs dpo on one metric, with a 0.5-threshold winner."""

    metric: str
    rs: RunSummary
    winner: RsDpoWinner
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "rs": self.rs.to_dict(),
            "winner": self.winner,
            "note": self.note,
        }


@dataclass
class RsDpoVerdict:
    """rs-sft vs dpo on identical prompts (model_b = rs in the h2h summary)."""

    rs_name: str
    dpo_name: str
    claim: str
    raw: RsDpoMetric
    length_controlled: RsDpoMetric | None
    primary_winner: RsDpoWinner
    primary_metric: str
    judge_bias: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rs_name": self.rs_name,
            "dpo_name": self.dpo_name,
            "claim": self.claim,
            "primary_metric": self.primary_metric,
            "primary_winner": self.primary_winner,
            "raw": self.raw.to_dict(),
            "length_controlled": (
                None
                if self.length_controlled is None
                else self.length_controlled.to_dict()
            ),
            "judge_bias": self.judge_bias,
        }

    def to_markdown(self) -> str:
        lines = [
            "# rs-sft vs dpo verdict (identical prompts)",
            "",
            f"- rs arm: `{self.rs_name}`",
            f"- dpo arm: `{self.dpo_name}`",
            f"- claim: {self.claim}",
            f"- primary metric: `{self.primary_metric}`",
            f"- primary winner: **{self.primary_winner}**",
            "",
            "## raw win-rate (rs vs dpo)",
            _rs_metric_md(self.raw),
        ]
        if self.length_controlled is not None:
            lines.extend(
                [
                    "",
                    "## length-controlled win-rate (rs vs dpo)",
                    _rs_metric_md(self.length_controlled),
                ]
            )
        lines.extend(["", "## judge bias", _judge_bias_md(self.judge_bias), ""])
        return "\n".join(lines)


def _rs_metric_md(m: RsDpoMetric) -> str:
    return "\n".join(
        [
            f"- rs win-rate: mean={m.rs.mean:.4f} ± std={m.rs.std:.4f} "
            f"ci95=[{m.rs.ci95_low:.4f}, {m.rs.ci95_high:.4f}] (n={m.rs.n})",
            f"- chance: 0.5",
            f"- winner: `{m.winner}` ({m.note})",
        ]
    )


def _judge_bias_md(bias: Mapping[str, Any] | None) -> str:
    if not bias:
        return "- not provided"
    pos = bias.get("position") or {}
    length = bias.get("length") or {}
    self_pref = bias.get("self_preference") or {}
    logprob = bias.get("logprob") or {}
    return "\n".join(
        [
            f"- position disagreement: {pos.get('disagreement_rate')}",
            f"- length-bias slope: {length.get('slope')}",
            f"- self-pref rate: {self_pref.get('self_pref_rate')} "
            f"(n_mixed={self_pref.get('n_mixed')})",
            f"- logprob agreement: {logprob.get('agreement_rate')}",
        ]
    )


def winner_against_chance(
    summary: RunSummary,
    *,
    chance: float = 0.5,
    ahead: RsDpoWinner = "rs_sft",
    behind: RsDpoWinner = "dpo",
) -> tuple[RsDpoWinner, str]:
    """if the mean ci excludes chance, declare a winner; else tie."""
    if summary.ci95_low > chance:
        return ahead, f"ci excludes {chance:g} ({ahead} ahead)"
    if summary.ci95_high < chance:
        return behind, f"ci excludes {chance:g} ({behind} ahead)"
    return "tie", f"ci includes {chance:g} (indistinguishable at 95%)"


def compare_rs_against_dpo(
    *,
    metric: str,
    rs_values: Sequence[float],
) -> RsDpoMetric:
    summary = summarize_runs(rs_values)
    winner, note = winner_against_chance(summary)
    return RsDpoMetric(metric=metric, rs=summary, winner=winner, note=note)


def build_rs_dpo_verdict(
    rs_vs_dpo: ArmVsSft,
    *,
    dpo_name: str,
    prefer_length_controlled: bool = True,
    judge_bias: Mapping[str, Any] | None = None,
) -> RsDpoVerdict:
    """compare rs vs dpo from a direct h2h (win_rate_b = rs)."""
    raw = compare_rs_against_dpo(
        metric="raw_win_rate_rs_vs_dpo",
        rs_values=rs_vs_dpo.win_rates_raw,
    )
    lc: RsDpoMetric | None = None
    if rs_vs_dpo.win_rates_lc:
        lc = compare_rs_against_dpo(
            metric="length_controlled_win_rate_rs_vs_dpo",
            rs_values=rs_vs_dpo.win_rates_lc,
        )
    if prefer_length_controlled and lc is not None:
        primary_metric = lc.metric
        primary_winner = lc.winner
    else:
        primary_metric = raw.metric
        primary_winner = raw.winner
    bias = dict(judge_bias) if judge_bias is not None else None
    return RsDpoVerdict(
        rs_name=rs_vs_dpo.name,
        dpo_name=dpo_name,
        claim=RS_DPO_CLAIM,
        raw=raw,
        length_controlled=lc,
        primary_winner=primary_winner,
        primary_metric=primary_metric,
        judge_bias=bias,
    )


def write_rs_dpo_verdict(
    verdict: RsDpoVerdict,
    path: str | Path | None = None,
    *,
    markdown_path: str | Path | None = None,
) -> Path:
    """write rs-vs-dpo verdict json (and optional markdown)."""
    out = DEFAULT_RS_VERDICT_PATH if path is None else resolve_path(path)
    write_json(out, verdict.to_dict())
    print(f"wrote rs-sft-vs-dpo verdict -> {out} winner={verdict.primary_winner}")
    if markdown_path is not None:
        md_out = resolve_path(markdown_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(verdict.to_markdown(), encoding="utf-8")
        print(f"wrote rs-sft-vs-dpo verdict markdown -> {md_out}")
    return out

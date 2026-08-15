"""did one training method beat another, and is the difference real?

two shapes of question, because the arms are not always judged the same way:

  DIRECT      rs-sft was judged head-to-head against dpo, so its win-rate already is
              the answer. parity is 0.5, and the question is whether the interval
              clears it. -> assess_head_to_head()

  INDIRECT    dpo and ppo were each judged against sft, never against each other. the
              answer is the difference between two win-rates over a shared opponent.
              -> compare_arms()

the interval comes from how many PROMPTS were judged, not from repeating the run.
generation and judging both run at temperature 0, so a repeat returns the identical
number — averaging repeats would report a standard deviation of zero and manufacture
confidence. a win-rate over n decisive pairs is a proportion, so the interval is the
binomial one.

ties leave the denominator. each pair is judged twice with the positions swapped, and a
disagreement is recorded as a tie — that means "no signal", not "half a win".
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.io import DEFAULT_METRICS_DIR, load_json_mapping, write_json
from prepare.paths import resolve_path

Z95 = 1.96
PARITY = 0.5
DEFAULT_VERDICT_PATH = DEFAULT_METRICS_DIR / "verdict.json"


@dataclass
class WinRate:
    """how one arm fared against a named opponent, in judged pairs."""

    arm: str        # the model this record is about, e.g. "ppo"
    opponent: str   # what it was judged against, e.g. "sft"
    wins: int       # pairs where the judge preferred `arm`
    losses: int     # pairs where the judge preferred `opponent`
    ties: int = 0   # pairs where the two position-swapped passes disagreed

    @property
    def decisive(self) -> int:
        """pairs that produced a signal; ties are not one."""
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        """share of decisive pairs won. 0.5 is parity with the opponent."""
        if self.decisive == 0:
            raise ValueError(f"{self.arm} vs {self.opponent}: every pair was a tie")
        return self.wins / self.decisive

    @property
    def stderr(self) -> float:
        """binomial standard error — shrinks with the number of prompts judged."""
        return math.sqrt(self.rate * (1 - self.rate) / self.decisive)

    @property
    def ci95(self) -> tuple[float, float]:
        half = Z95 * self.stderr
        return self.rate - half, self.rate + half

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci95
        return {
            "arm": self.arm,
            "opponent": self.opponent,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "n_decisive": self.decisive,
            "rate": self.rate,
            "ci95_low": lo,
            "ci95_high": hi,
        }


@dataclass
class Verdict:
    """which arm won, by how much, and whether the margin survives its interval."""

    # one line naming what was compared, for the top of the report. generated from the
    # arm names unless the caller passes something better.
    question: str

    # the arm on trial. in a direct head-to-head this is the only record there is.
    challenger: WinRate

    # the arm it must beat. None for a direct head-to-head, where the opponent is
    # already baked into `challenger` and the bar is parity rather than another arm.
    baseline: WinRate | None

    # the challenger's advantage. against a baseline: its win-rate minus the
    # baseline's. in a direct head-to-head: how far above 0.5 it landed.
    # positive always favours the challenger.
    margin: float

    # 95% interval on `margin`. a winner is called only when this excludes zero —
    # if it straddles zero the margin is inside the noise.
    ci95_low: float
    ci95_high: float

    # the NAME of the winning arm, or None when the interval straddles zero.
    # None means "we could not tell", not "they are equal".
    winner: str | None

    # plain-language why, carried into the report so the number is not read alone.
    reason: str

    @property
    def decided(self) -> bool:
        return self.winner is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "challenger": self.challenger.to_dict(),
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "margin": self.margin,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "winner": self.winner,
            "decided": self.decided,
            "reason": self.reason,
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", ""]
        for arm in (self.challenger, self.baseline):
            if arm is None:
                continue
            lo, hi = arm.ci95
            lines.append(
                f"- `{arm.arm}` vs `{arm.opponent}`: {arm.rate:.3f} "
                f"ci95=[{lo:.3f}, {hi:.3f}] "
                f"({arm.wins}W / {arm.losses}L / {arm.ties}T over {arm.decisive} decisive)"
            )
        outcome = f"**{self.winner}**" if self.winner else "**no winner**"
        lines += [
            "",
            f"- margin: {self.margin:+.3f} "
            f"ci95=[{self.ci95_low:+.3f}, {self.ci95_high:+.3f}]",
            f"- verdict: {outcome} — {self.reason}",
            "",
        ]
        return "\n".join(lines)


def assess_head_to_head(result: WinRate, *, question: str | None = None) -> Verdict:
    """did this arm beat the opponent it was judged against?

    the margin is how far the win-rate sits above parity. a winner is called only when
    the whole interval clears 0.5 — a rate above 0.5 with an interval that still touches
    it is run-to-run noise, not a result.
    """
    margin = result.rate - PARITY
    half = Z95 * result.stderr
    lo, hi = margin - half, margin + half
    if lo > 0:
        winner, reason = result.arm, f"interval clears parity ({result.arm} ahead)"
    elif hi < 0:
        winner, reason = result.opponent, f"interval clears parity ({result.opponent} ahead)"
    else:
        winner, reason = None, "interval includes parity; indistinguishable at 95%"
    return Verdict(
        question=question or f"{result.arm} vs {result.opponent}",
        challenger=result,
        baseline=None,
        margin=margin,
        ci95_low=lo,
        ci95_high=hi,
        winner=winner,
        reason=reason,
    )


def compare_arms(
    challenger: WinRate,
    baseline: WinRate,
    *,
    question: str | None = None,
) -> Verdict:
    """which of two arms is better, given both were judged against the same opponent?

    the margin is the challenger's win-rate minus the baseline's. both must share an
    opponent — comparing `dpo vs sft` against `ppo vs base` would be measuring two
    different things and reporting the difference as if it meant something.
    """
    if challenger.opponent != baseline.opponent:
        raise ValueError(
            f"arms were judged against different opponents "
            f"({challenger.arm} vs {challenger.opponent}, "
            f"{baseline.arm} vs {baseline.opponent}); not comparable"
        )
    margin = challenger.rate - baseline.rate
    half = Z95 * math.sqrt(challenger.stderr**2 + baseline.stderr**2)
    lo, hi = margin - half, margin + half
    if lo > 0:
        winner, reason = challenger.arm, f"interval excludes 0 ({challenger.arm} ahead)"
    elif hi < 0:
        winner, reason = baseline.arm, f"interval excludes 0 ({baseline.arm} ahead)"
    else:
        winner, reason = None, "interval includes 0; indistinguishable at 95%"
    return Verdict(
        question=question or f"{challenger.arm} vs {baseline.arm} (both vs {baseline.opponent})",
        challenger=challenger,
        baseline=baseline,
        margin=margin,
        ci95_low=lo,
        ci95_high=hi,
        winner=winner,
        reason=reason,
    )


def load_win_rate(
    summary: str | Path | Mapping[str, Any],
    *,
    arm: str | None = None,
    opponent: str | None = None,
) -> WinRate:
    """read a head-to-head summary. model b is the arm, model a is the opponent.

    counts are summed across every report in the file, so a summary holding more than
    one pass is treated as one larger sample rather than an average of small ones.
    """
    payload = load_json_mapping(summary)
    reports = payload.get("reports") or []
    if not reports:
        raise ValueError("head-to-head summary has no reports")

    arm_name = arm or Path(str(payload.get("model_b") or "model_b")).name
    opponent_name = opponent or Path(str(payload.get("model_a") or "model_a")).name

    wins = losses = ties = 0
    for report in reports:
        raw = report.get("raw") or {}
        wins += int(raw.get("wins_b") or 0)
        losses += int(raw.get("wins_a") or 0)
        ties += int(raw.get("ties") or 0)
    if wins + losses == 0:
        raise ValueError(f"{arm_name} vs {opponent_name}: every judged pair was a tie")
    return WinRate(
        arm=arm_name, opponent=opponent_name, wins=wins, losses=losses, ties=ties
    )


def write_verdict(
    verdict: Verdict,
    path: str | Path | None = None,
    *,
    markdown_path: str | Path | None = None,
) -> Path:
    """write the verdict json, and markdown alongside it if asked."""
    out = DEFAULT_VERDICT_PATH if path is None else resolve_path(path)
    write_json(out, verdict.to_dict())
    print(f"wrote verdict -> {out} winner={verdict.winner or 'none'}")
    if markdown_path is not None:
        md = resolve_path(markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(verdict.to_markdown(), encoding="utf-8")
        print(f"wrote verdict markdown -> {md}")
    return out

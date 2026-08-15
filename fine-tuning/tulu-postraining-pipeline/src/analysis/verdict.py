"""judged-win-rate verdicts: dpo vs ppo, and rs-sft against either.

the uncertainty comes from how many PROMPTS were judged, not from repeating the run.
generation and judging both run at temperature 0, so a repeat returns the identical
number — averaging repeats would report a standard deviation of zero and manufacture
confidence. a win-rate over n decisive pairs is a proportion, and its interval is the
binomial one.

ties are excluded from the denominator: the judge is asked twice with the positions
swapped and a disagreement is recorded as a tie, so ties mean "no signal", not "half a
win".
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from analysis.io import DEFAULT_METRICS_DIR, load_json_mapping, write_json
from prepare.paths import resolve_path

Z95 = 1.96
CHANCE = 0.5
Winner = Literal["a", "b", "tie"]
DEFAULT_VERDICT_PATH = DEFAULT_METRICS_DIR / "verdict.json"


@dataclass
class WinRate:
    """one arm's judged record against a common opponent."""

    name: str
    wins: int
    losses: int
    ties: int = 0

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        if self.decisive == 0:
            raise ValueError(f"{self.name}: no decisive pairs, only ties")
        return self.wins / self.decisive

    @property
    def stderr(self) -> float:
        p, n = self.rate, self.decisive
        return math.sqrt(p * (1 - p) / n)

    @property
    def ci95(self) -> tuple[float, float]:
        half = Z95 * self.stderr
        return self.rate - half, self.rate + half

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci95
        return {
            "name": self.name,
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
    """the comparison and its confidence interval."""

    question: str
    a: WinRate
    b: WinRate | None
    delta: float
    ci95_low: float
    ci95_high: float
    winner: Winner
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "a": self.a.to_dict(),
            "b": None if self.b is None else self.b.to_dict(),
            "delta": self.delta,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "winner": self.winner,
            "note": self.note,
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", ""]
        for arm in (self.a, self.b):
            if arm is None:
                continue
            lo, hi = arm.ci95
            lines.append(
                f"- `{arm.name}`: {arm.rate:.3f} "
                f"ci95=[{lo:.3f}, {hi:.3f}] "
                f"({arm.wins}W / {arm.losses}L / {arm.ties}T)"
            )
        lines += [
            "",
            f"- delta: {self.delta:+.3f} ci95=[{self.ci95_low:+.3f}, {self.ci95_high:+.3f}]",
            f"- winner: **{self.winner}** ({self.note})",
            "",
        ]
        return "\n".join(lines)


def _call(delta: float, lo: float, hi: float, a: str, b: str) -> tuple[Winner, str]:
    if lo > 0:
        return "b", f"ci excludes 0 ({b} ahead)"
    if hi < 0:
        return "a", f"ci excludes 0 ({a} ahead)"
    return "tie", "ci includes 0 (indistinguishable at 95%)"


def compare(a: WinRate, b: WinRate, *, question: str) -> Verdict:
    """two arms judged against a common opponent; interval on the difference."""
    delta = b.rate - a.rate
    half = Z95 * math.sqrt(a.stderr**2 + b.stderr**2)
    lo, hi = delta - half, delta + half
    winner, note = _call(delta, lo, hi, a.name, b.name)
    return Verdict(
        question=question,
        a=a,
        b=b,
        delta=delta,
        ci95_low=lo,
        ci95_high=hi,
        winner=winner,
        note=note,
    )


def compare_to_chance(arm: WinRate, *, question: str, chance: float = CHANCE) -> Verdict:
    """one arm judged directly against an opponent, so chance is 0.5 by construction."""
    delta = arm.rate - chance
    half = Z95 * arm.stderr
    lo, hi = delta - half, delta + half
    if lo > 0:
        winner, note = "b", f"ci excludes {chance:g} ({arm.name} ahead)"
    elif hi < 0:
        winner, note = "a", f"ci excludes {chance:g} (opponent ahead)"
    else:
        winner, note = "tie", f"ci includes {chance:g} (indistinguishable at 95%)"
    return Verdict(
        question=question,
        a=arm,
        b=None,
        delta=delta,
        ci95_low=lo,
        ci95_high=hi,
        winner=winner,
        note=note,
    )


def load_win_rate(name: str, summary: str | Path | Mapping[str, Any]) -> WinRate:
    """read a head-to-head summary; model b is the named arm."""
    payload = load_json_mapping(summary)
    reports = payload.get("reports") or []
    if not reports:
        raise ValueError(f"{name}: head-to-head summary has no reports")
    wins = losses = ties = 0
    for report in reports:
        raw = report.get("raw") or {}
        wins += int(raw.get("wins_b") or 0)
        losses += int(raw.get("wins_a") or 0)
        ties += int(raw.get("ties") or 0)
    if wins + losses == 0:
        raise ValueError(f"{name}: every judged pair was a tie")
    return WinRate(name=name, wins=wins, losses=losses, ties=ties)


def write_verdict(
    verdict: Verdict,
    path: str | Path | None = None,
    *,
    markdown_path: str | Path | None = None,
) -> Path:
    """write the verdict json, and markdown alongside it if asked."""
    out = DEFAULT_VERDICT_PATH if path is None else resolve_path(path)
    write_json(out, verdict.to_dict())
    print(f"wrote verdict -> {out} winner={verdict.winner}")
    if markdown_path is not None:
        md = resolve_path(markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(verdict.to_markdown(), encoding="utf-8")
        print(f"wrote verdict markdown -> {md}")
    return out

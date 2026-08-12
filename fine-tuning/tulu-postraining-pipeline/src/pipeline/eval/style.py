"""length / markdown style metrics and length-controlled head-to-head win-rates.

length control keeps pairs whose completions are similar in length
(|len_a - len_b| / max(len) <= max_rel_diff), then recomputes win-rate on
that subset. raw win-rate uses the full judge set. both are reported.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.eval.io import load_jsonl

# common markdown tells (headers, emphasis, lists, fences, links)
_MARKDOWN_PATTERNS = (
    re.compile(r"(?m)^#{1,6}\s"),  # headers
    re.compile(r"\*\*[^*]+\*\*|__[^_]+__"),  # bold
    re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_)"),  # italic
    re.compile(r"(?m)^\s*[-*+]\s+\S"),  # unordered list
    re.compile(r"(?m)^\s*\d+\.\s+\S"),  # ordered list
    re.compile(r"```"),  # fenced code
    re.compile(r"\[[^\]]+\]\([^)]+\)"),  # links
    re.compile(r"`[^`]+`"),  # inline code
)

DEFAULT_MAX_REL_LENGTH_DIFF = 0.10


@dataclass
class StyleSummary:
    """aggregate length / markdown stats for a set of completions."""

    n: int
    mean_chars: float
    mean_words: float
    markdown_rate: float
    mean_markdown_hits: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WinRateSummary:
    """head-to-head win-rate for model B vs model A (ties counted separately)."""

    n: int
    wins_a: int
    wins_b: int
    ties: int
    # win_rate_b among decisive (non-tie) pairs; none if no decisive pairs
    win_rate_b: float | None
    # with ties as 0.5 share for B
    win_rate_b_with_ties: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeadToHeadStyleReport:
    """raw + length-controlled win-rates and per-side style stats."""

    n_total: int
    n_length_matched: int
    max_rel_length_diff: float
    style_a: StyleSummary
    style_b: StyleSummary
    raw: WinRateSummary
    length_controlled: WinRateSummary
    mean_char_delta_b_minus_a: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_length_matched": self.n_length_matched,
            "max_rel_length_diff": self.max_rel_length_diff,
            "style_a": self.style_a.to_dict(),
            "style_b": self.style_b.to_dict(),
            "raw": self.raw.to_dict(),
            "length_controlled": self.length_controlled.to_dict(),
            "mean_char_delta_b_minus_a": self.mean_char_delta_b_minus_a,
        }


def char_length(text: str) -> int:
    return len(text or "")


def word_length(text: str) -> int:
    return len((text or "").split())


def markdown_hit_count(text: str) -> int:
    """count how many markdown pattern families fire (0..len(patterns))."""
    s = text or ""
    return sum(1 for pat in _MARKDOWN_PATTERNS if pat.search(s))


def has_markdown(text: str) -> bool:
    return markdown_hit_count(text) > 0


def summarize_style(completions: Sequence[str]) -> StyleSummary:
    """mean length + markdown rate over completions."""
    n = len(completions)
    if n == 0:
        return StyleSummary(
            n=0,
            mean_chars=0.0,
            mean_words=0.0,
            markdown_rate=0.0,
            mean_markdown_hits=0.0,
        )
    chars = [char_length(c) for c in completions]
    words = [word_length(c) for c in completions]
    hits = [markdown_hit_count(c) for c in completions]
    md = [1 if h > 0 else 0 for h in hits]
    return StyleSummary(
        n=n,
        mean_chars=sum(chars) / n,
        mean_words=sum(words) / n,
        markdown_rate=sum(md) / n,
        mean_markdown_hits=sum(hits) / n,
    )


def relative_length_diff(len_a: int, len_b: int) -> float:
    """|a-b| / max(a,b); 0 if both empty."""
    denom = max(len_a, len_b)
    if denom == 0:
        return 0.0
    return abs(len_a - len_b) / denom


def is_length_matched(
    text_a: str,
    text_b: str,
    *,
    max_rel_diff: float = DEFAULT_MAX_REL_LENGTH_DIFF,
) -> bool:
    return relative_length_diff(char_length(text_a), char_length(text_b)) <= max_rel_diff


def compute_win_rates(records: Sequence[Mapping[str, Any]]) -> WinRateSummary:
    """aggregate winner field (A/B/tie) into win-rates for B."""
    wins_a = wins_b = ties = 0
    for row in records:
        w = row.get("winner")
        if w == "A":
            wins_a += 1
        elif w == "B":
            wins_b += 1
        else:
            ties += 1
    n = wins_a + wins_b + ties
    decisive = wins_a + wins_b
    win_rate_b = (wins_b / decisive) if decisive else None
    win_rate_b_with_ties = ((wins_b + 0.5 * ties) / n) if n else None
    return WinRateSummary(
        n=n,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_b=win_rate_b,
        win_rate_b_with_ties=win_rate_b_with_ties,
    )


def report_head_to_head_style(
    records: Sequence[Mapping[str, Any]],
    *,
    max_rel_length_diff: float = DEFAULT_MAX_REL_LENGTH_DIFF,
) -> HeadToHeadStyleReport:
    """raw + length-controlled win-rates and style stats from judge records."""
    if max_rel_length_diff < 0:
        raise ValueError(f"max_rel_length_diff must be >= 0, got {max_rel_length_diff}")

    completions_a: list[str] = []
    completions_b: list[str] = []
    matched: list[Mapping[str, Any]] = []
    deltas: list[float] = []

    for row in records:
        a = str(row.get("completion_a") or "")
        b = str(row.get("completion_b") or "")
        completions_a.append(a)
        completions_b.append(b)
        deltas.append(float(char_length(b) - char_length(a)))
        if is_length_matched(a, b, max_rel_diff=max_rel_length_diff):
            matched.append(row)

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return HeadToHeadStyleReport(
        n_total=len(records),
        n_length_matched=len(matched),
        max_rel_length_diff=max_rel_length_diff,
        style_a=summarize_style(completions_a),
        style_b=summarize_style(completions_b),
        raw=compute_win_rates(records),
        length_controlled=compute_win_rates(matched),
        mean_char_delta_b_minus_a=mean_delta,
    )


def report_head_to_head_style_from_jsonl(
    path: str | Path,
    *,
    max_rel_length_diff: float = DEFAULT_MAX_REL_LENGTH_DIFF,
) -> HeadToHeadStyleReport:
    """load judge jsonl then report raw + length-controlled metrics."""
    return report_head_to_head_style(
        load_jsonl(path),
        max_rel_length_diff=max_rel_length_diff,
    )

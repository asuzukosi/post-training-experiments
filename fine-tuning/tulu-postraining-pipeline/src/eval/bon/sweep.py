"""best-of-n sweep: proxy select at 8 n-values, gold-score vs n=1, kl ≈ log n."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.bon.candidates import COMPLETION_KEY, PROMPT_ID_KEY, group_candidates
from eval.bon.proxy import (
    LOGPROB_KEY,
    PROXY_SCORE_KEY,
    proxy_score,
    select_top1_by_proxy,
)
from eval.bon.tournament import sort_pairs_by_length
from eval.io import ID_KEY, PROMPT_KEY, append_jsonl, load_jsonl
from eval.judge import (
    DEFAULT_JUDGE_BATCH_SIZE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    judge_incremental,
)
from eval.style import (
    DEFAULT_MAX_REL_LENGTH_DIFF,
    char_length,
    report_head_to_head_style,
)
from prepare.paths import resolve_path

DEFAULT_N_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
BASELINE_N = 1


@dataclass
class SweepPoint:
    """one n on the over-optimization curve."""

    n: int
    n_prompts: int
    mean_proxy: float
    kl: float
    gold_win_rate: float | None
    gold_win_rate_lc: float | None
    mean_chars: float
    mean_logprob: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepReport:
    n_values: list[int]
    n_prompts: int
    points: list[SweepPoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_values": list(self.n_values),
            "n_prompts": self.n_prompts,
            "points": [p.to_dict() for p in self.points],
        }


def parse_n_values(raw: Sequence[int] | str) -> list[int]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(v) for v in raw]
    if not values:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in values):
        raise ValueError(f"n_values must be >= 1, got {values}")
    return sorted(set(values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _optional_logprob(row: Mapping[str, Any]) -> float | None:
    if LOGPROB_KEY not in row or row[LOGPROB_KEY] is None:
        return None
    value = float(row[LOGPROB_KEY])
    if value != value:
        return None
    return value


def _summarize_winners(winners: Mapping[str, Mapping[str, Any]], n: int) -> SweepPoint:
    rows = [winners[pid] for pid in sorted(winners)]
    logprobs = [_optional_logprob(r) for r in rows]
    have_lp = [v for v in logprobs if v is not None]
    return SweepPoint(
        n=n,
        n_prompts=len(rows),
        mean_proxy=_mean([proxy_score(r) for r in rows]),
        kl=math.log(n),
        gold_win_rate=None,
        gold_win_rate_lc=None,
        mean_chars=_mean([float(char_length(str(r[COMPLETION_KEY]))) for r in rows]),
        mean_logprob=_mean(have_lp) if have_lp else None,
    )


def build_gold_pairs(
    baseline: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Mapping[str, Any]],
    *,
    n: int,
) -> list[dict[str, Any]]:
    """n=1 completions as a vs selected-at-n as b."""
    missing = [pid for pid in baseline if pid not in selected]
    extra = [pid for pid in selected if pid not in baseline]
    if missing or extra:
        raise ValueError(
            f"baseline/selected prompt_id mismatch missing={missing} extra={extra}"
        )
    pairs: list[dict[str, Any]] = []
    for prompt_id in sorted(baseline):
        a = baseline[prompt_id]
        b = selected[prompt_id]
        pairs.append(
            {
                ID_KEY: f"{prompt_id}__n{n}",
                PROMPT_ID_KEY: prompt_id,
                PROMPT_KEY: a[PROMPT_KEY],
                "completion_a": a[COMPLETION_KEY],
                "completion_b": b[COMPLETION_KEY],
                "model_a": f"bon_n{BASELINE_N}",
                "model_b": f"bon_n{n}",
                "n": n,
            }
        )
    return pairs


def _write_selections(
    path: Path,
    winners: Mapping[str, Mapping[str, Any]],
    *,
    n: int,
) -> None:
    if path.exists():
        path.unlink()
    for prompt_id in sorted(winners):
        append_jsonl(path, {**dict(winners[prompt_id]), "n": n})


def run_bon_sweep(
    *,
    generations_path: str | Path,
    output_dir: str | Path,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    max_rel_length_diff: float = DEFAULT_MAX_REL_LENGTH_DIFF,
) -> SweepReport:
    """proxy select at each n; gold-score vs n=1; write sweep.json."""
    ns = parse_n_values(n_values)
    gens = load_jsonl(generations_path)
    if not gens:
        raise ValueError(f"no generations in {generations_path}")
    grouped = group_candidates(gens)
    for prompt_id, cands in grouped.items():
        for row in cands:
            proxy_score(row)

    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = select_top1_by_proxy(grouped, BASELINE_N)
    points: list[SweepPoint] = []

    for n in ns:
        winners = select_top1_by_proxy(grouped, n)
        n_dir = out_dir / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)
        _write_selections(n_dir / "selections.jsonl", winners, n=n)
        point = _summarize_winners(winners, n)
        if n == BASELINE_N:
            point.gold_win_rate = 0.5
            point.gold_win_rate_lc = 0.5
        else:
            pairs = sort_pairs_by_length(
                build_gold_pairs(baseline, winners, n=n)
            )
            judge_path = n_dir / "judge.jsonl"
            judge_incremental(
                pairs,
                judge_model=judge_model,
                output_path=judge_path,
                temperature=temperature,
                batch_size=batch_size,
            )
            report = report_head_to_head_style(
                load_jsonl(judge_path),
                max_rel_length_diff=max_rel_length_diff,
            )
            point.gold_win_rate = report.raw.win_rate_b_with_ties
            point.gold_win_rate_lc = report.length_controlled.win_rate_b_with_ties
        points.append(point)
        print(
            f"bon sweep: n={n} mean_proxy={point.mean_proxy} "
            f"kl={point.kl} gold={point.gold_win_rate} "
            f"gold_lc={point.gold_win_rate_lc}"
        )

    report = SweepReport(
        n_values=ns,
        n_prompts=len(grouped),
        points=points,
    )
    summary_path = out_dir / "sweep.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
        f.write("\n")
    print(f"bon sweep done: n_values={ns} wrote={summary_path}")
    return report

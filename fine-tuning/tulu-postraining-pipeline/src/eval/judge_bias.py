"""judge-reliability metrics from judgearena records.

position: disagreement rate between order_ab and order_ba.
length: ols slope of p(b wins) on char_len(b) - char_len(a).
self-pref: how often the judge picks the qwen side vs smollm/olmo.
logprob: optional diagnostic — agreement between avg answer logprob and winner.
not a replacement judge.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.io import load_jsonl
from eval.judge import FIRST_MODEL, MODEL_TIE, SECOND_MODEL
from eval.style import _require_judge_record, char_length

QWEN_TAG = "qwen"
PROBE_TAGS = ("smollm", "olmo")
LOGPROB_A_KEY = "avg_logprob_a"
LOGPROB_B_KEY = "avg_logprob_b"


@dataclass
class PositionBias:
    n: int
    n_disagree: int
    disagreement_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LengthBias:
    n: int
    slope: float | None
    intercept: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfPreference:
    n_mixed: int
    n_qwen_wins: int
    n_probe_wins: int
    n_ties: int
    self_pref_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogprobDiagnostic:
    n: int
    n_agree: int
    agreement_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeBiasReport:
    n: int
    position: PositionBias
    length: LengthBias
    self_preference: SelfPreference
    logprob: LogprobDiagnostic

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "position": self.position.to_dict(),
            "length": self.length.to_dict(),
            "self_preference": self.self_preference.to_dict(),
            "logprob": self.logprob.to_dict(),
        }


def is_qwen_family(model: str | None) -> bool:
    return QWEN_TAG in (model or "").lower()


def is_probe_family(model: str | None) -> bool:
    low = (model or "").lower()
    return any(tag in low for tag in PROBE_TAGS)


def _p_b_wins(row: Mapping[str, Any], winner: str) -> float:
    pref = row.get("pref_ab")
    if pref is not None:
        value = float(pref)
        if value == value:
            return value
    if winner == SECOND_MODEL:
        return 1.0
    if winner == FIRST_MODEL:
        return 0.0
    return 0.5


def _require_order(row: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("order_ab", "order_ba"):
        if key not in row:
            raise ValueError(f"judge record missing {key!r} (id={row.get('id')!r})")
        val = row[key]
        if val not in (FIRST_MODEL, SECOND_MODEL, MODEL_TIE):
            raise ValueError(
                f"judge record {key!r} must be A/B/tie, got {val!r} "
                f"(id={row.get('id')!r})"
            )
    return str(row["order_ab"]), str(row["order_ba"])


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float | None, float | None]:
    n = len(xs)
    if n < 2:
        return None, None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None, mean_y
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    value = float(row[key])
    if value != value:
        return None
    return value


def position_disagreement(records: Sequence[Mapping[str, Any]]) -> PositionBias:
    n_disagree = 0
    for row in records:
        order_ab, order_ba = _require_order(row)
        if order_ab != order_ba:
            n_disagree += 1
    n = len(records)
    rate = (n_disagree / n) if n else None
    return PositionBias(n=n, n_disagree=n_disagree, disagreement_rate=rate)


def length_bias_slope(records: Sequence[Mapping[str, Any]]) -> LengthBias:
    xs: list[float] = []
    ys: list[float] = []
    for row in records:
        completion_a, completion_b, winner = _require_judge_record(row)
        xs.append(float(char_length(completion_b) - char_length(completion_a)))
        ys.append(_p_b_wins(row, winner))
    slope, intercept = _ols(xs, ys)
    return LengthBias(n=len(records), slope=slope, intercept=intercept)


def self_preference_rate(records: Sequence[Mapping[str, Any]]) -> SelfPreference:
    n_qwen = n_probe = n_ties = 0
    n_mixed = 0
    for row in records:
        _require_judge_record(row)
        model_a = str(row.get("model_a") or "")
        model_b = str(row.get("model_b") or "")
        a_qwen = is_qwen_family(model_a)
        b_qwen = is_qwen_family(model_b)
        a_probe = is_probe_family(model_a)
        b_probe = is_probe_family(model_b)
        if a_qwen and b_probe:
            qwen_side = FIRST_MODEL
        elif b_qwen and a_probe:
            qwen_side = SECOND_MODEL
        else:
            continue
        n_mixed += 1
        winner = row["winner"]
        if winner == MODEL_TIE:
            n_ties += 1
        elif winner == qwen_side:
            n_qwen += 1
        else:
            n_probe += 1
    n = n_mixed
    rate = ((n_qwen + 0.5 * n_ties) / n) if n else None
    return SelfPreference(
        n_mixed=n_mixed,
        n_qwen_wins=n_qwen,
        n_probe_wins=n_probe,
        n_ties=n_ties,
        self_pref_rate=rate,
    )


def logprob_agreement(records: Sequence[Mapping[str, Any]]) -> LogprobDiagnostic:
    n_agree = 0
    n = 0
    for row in records:
        _, _, winner = _require_judge_record(row)
        if winner == MODEL_TIE:
            continue
        log_a = _optional_float(row, LOGPROB_A_KEY)
        log_b = _optional_float(row, LOGPROB_B_KEY)
        if log_a is None or log_b is None or log_a == log_b:
            continue
        n += 1
        logprob_picks_b = log_b > log_a
        judge_picks_b = winner == SECOND_MODEL
        if logprob_picks_b == judge_picks_b:
            n_agree += 1
    rate = (n_agree / n) if n else None
    return LogprobDiagnostic(n=n, n_agree=n_agree, agreement_rate=rate)


def report_judge_bias(records: Sequence[Mapping[str, Any]]) -> JudgeBiasReport:
    """compute position / length / self-pref / logprob metrics from judge records."""
    for row in records:
        _require_judge_record(row)
        _require_order(row)
    return JudgeBiasReport(
        n=len(records),
        position=position_disagreement(records),
        length=length_bias_slope(records),
        self_preference=self_preference_rate(records),
        logprob=logprob_agreement(records),
    )


def report_judge_bias_from_jsonl(path: str | Path) -> JudgeBiasReport:
    return report_judge_bias(load_jsonl(path))

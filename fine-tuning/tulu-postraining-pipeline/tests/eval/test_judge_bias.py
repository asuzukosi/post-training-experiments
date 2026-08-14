"""unit tests for judge-reliability metrics. no gpu."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.io import append_jsonl
from eval.judge import FIRST_MODEL, MODEL_TIE, SECOND_MODEL
from eval.judge_bias import (
    report_judge_bias,
    report_judge_bias_from_jsonl,
)

QWEN = "Qwen/Qwen2.5-32B-Instruct"
SMOL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
OLMO = "allenai/OLMo-2-0425-1B-Instruct"


def _record(
    item_id: str,
    completion_a: str,
    completion_b: str,
    winner: str,
    *,
    order_ab: str | None = None,
    order_ba: str | None = None,
    pref_ab: float | None = None,
    model_a: str = QWEN,
    model_b: str = QWEN,
    avg_logprob_a: float | None = None,
    avg_logprob_b: float | None = None,
) -> dict:
    if pref_ab is None:
        pref_ab = {"A": 0.0, "B": 1.0, "tie": 0.5}[winner]
    if order_ab is None:
        order_ab = winner
    if order_ba is None:
        order_ba = winner
    row: dict = {
        "id": item_id,
        "completion_a": completion_a,
        "completion_b": completion_b,
        "winner": winner,
        "order_ab": order_ab,
        "order_ba": order_ba,
        "pref_ab": pref_ab,
        "model_a": model_a,
        "model_b": model_b,
        "judge_backend": "judgearena",
    }
    if avg_logprob_a is not None:
        row["avg_logprob_a"] = avg_logprob_a
    if avg_logprob_b is not None:
        row["avg_logprob_b"] = avg_logprob_b
    return row


def test_position_disagreement_rate() -> None:
    report = report_judge_bias(
        [
            _record("0", "aa", "bb", FIRST_MODEL),
            _record(
                "1",
                "aa",
                "bb",
                MODEL_TIE,
                order_ab=FIRST_MODEL,
                order_ba=SECOND_MODEL,
            ),
            _record(
                "2",
                "aa",
                "bb",
                MODEL_TIE,
                order_ab=SECOND_MODEL,
                order_ba=FIRST_MODEL,
            ),
        ]
    )
    assert report.position.n == 3
    assert report.position.n_disagree == 2
    assert report.position.disagreement_rate == pytest.approx(2 / 3)


def test_length_bias_slope_prefers_longer() -> None:
    # longer b wins; longer a loses → positive slope of p(b) vs len_b-len_a
    report = report_judge_bias(
        [
            _record("0", "x", "yyyyyyyy", SECOND_MODEL),
            _record("1", "xxxxxxxx", "y", FIRST_MODEL),
        ]
    )
    assert report.length.n == 2
    assert report.length.slope is not None
    assert report.length.slope > 0


def test_self_preference_vs_probe_families() -> None:
    report = report_judge_bias(
        [
            _record("0", "qwen a", "smol b", FIRST_MODEL, model_a=QWEN, model_b=SMOL),
            _record("1", "olmo a", "qwen b", SECOND_MODEL, model_a=OLMO, model_b=QWEN),
            _record("2", "qwen a", "olmo b", MODEL_TIE, model_a=QWEN, model_b=OLMO),
            # same-family pair is ignored for self-pref
            _record("3", "sft", "dpo", SECOND_MODEL, model_a=QWEN, model_b=QWEN),
        ]
    )
    sp = report.self_preference
    assert sp.n_mixed == 3
    assert sp.n_qwen_wins == 2
    assert sp.n_probe_wins == 0
    assert sp.n_ties == 1
    assert sp.self_pref_rate == pytest.approx((2 + 0.5) / 3)



def test_report_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    append_jsonl(
        path,
        _record("0", "aa", "bbbb", SECOND_MODEL, model_a=QWEN, model_b=SMOL),
    )
    report = report_judge_bias_from_jsonl(path)
    assert report.n == 1
    assert report.position.disagreement_rate == 0.0
    assert report.self_preference.n_mixed == 1



"""unit tests for rewardbench-chat gate thresholds (no model required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.rb_gate_chat import (
    DEFAULT_MIN_ACC,
    DEFAULT_WARN_ACC,
    GateResult,
    GateStatus,
    decide_gate,
)


def test_decide_gate_pass_warn_fail() -> None:
    assert decide_gate(0.71) is GateStatus.PASS
    assert decide_gate(0.70) is GateStatus.PASS
    assert decide_gate(0.69) is GateStatus.WARN
    assert decide_gate(0.65) is GateStatus.WARN
    assert decide_gate(0.649) is GateStatus.FAIL
    assert decide_gate(0.0) is GateStatus.FAIL


def test_decide_gate_custom_thresholds() -> None:
    assert decide_gate(0.66, min_acc=0.60, warn_acc=0.68) is GateStatus.WARN
    assert decide_gate(0.68, min_acc=0.60, warn_acc=0.68) is GateStatus.PASS
    assert decide_gate(0.59, min_acc=0.60, warn_acc=0.68) is GateStatus.FAIL


def test_decide_gate_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="min_acc"):
        decide_gate(0.7, min_acc=0.8, warn_acc=0.7)


def test_gate_result_raise_if_failed() -> None:
    ok = GateResult(
        chat_accuracy=0.66,
        n=100,
        status=GateStatus.WARN,
        min_acc=DEFAULT_MIN_ACC,
        warn_acc=DEFAULT_WARN_ACC,
    )
    ok.raise_if_failed()  # warn does not raise

    bad = GateResult(
        chat_accuracy=0.5,
        n=100,
        status=GateStatus.FAIL,
        min_acc=DEFAULT_MIN_ACC,
        warn_acc=DEFAULT_WARN_ACC,
    )
    with pytest.raises(RuntimeError, match="gate failed"):
        bad.raise_if_failed()

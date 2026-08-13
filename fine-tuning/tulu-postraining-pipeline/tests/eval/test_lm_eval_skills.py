"""unit tests for lm-eval skills wrapper (no harness / no gpu).

`run_skills_eval` has a single execution path into `lm_eval.simple_evaluate`, so
these tests patch `sys.modules["lm_eval"]` rather than injecting a callable. that
keeps `_default_simple_evaluate`'s kwargs assembly under test — it is the only
place the lm-eval contract is expressed, and it is what a real run depends on.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.lm_eval_skills import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_BACKEND,
    extract_ifeval_score,
    extract_MMLU_acc,
    extract_task_metrics,
    flag_MMLU_drop,
    run_skills_eval,
)


@pytest.fixture
def fake_lm_eval(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """stand in for the lm-eval package; record what simple_evaluate receives.

    monkeypatch.setitem restores sys.modules afterwards, so this works whether or
    not lm-eval is actually installed (it is not, in the local dev venv).
    """
    recorder = SimpleNamespace(
        calls=[],
        results={
            "results": {
                "ifeval": {"prompt_level_strict_acc,none": 0.55},
                "mmlu": {"acc,none": 0.30},
            }
        },
    )

    def simple_evaluate(**kwargs: object) -> dict:
        recorder.calls.append(kwargs)
        return recorder.results

    module = types.ModuleType("lm_eval")
    module.simple_evaluate = simple_evaluate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lm_eval", module)
    return recorder


def test_flag_MMLU_drop_threshold() -> None:
    ok = flag_MMLU_drop(0.50, 0.46)  # 4 pts
    assert ok.flagged is False
    assert ok.drop_pts == pytest.approx(4.0)

    bad = flag_MMLU_drop(0.50, 0.44)  # 6 pts
    assert bad.flagged is True
    assert bad.drop_pts == pytest.approx(6.0)

    edge = flag_MMLU_drop(0.50, 0.45)  # exactly 5 pts -> not > 5
    assert edge.flagged is False


def test_flag_MMLU_drop_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="baseline_acc"):
        flag_MMLU_drop(1.5, 0.4)
    with pytest.raises(ValueError, match="threshold"):
        flag_MMLU_drop(0.5, 0.4, threshold_pts=-1)


def test_extract_metrics_from_lm_eval_shape() -> None:
    raw = {
        "results": {
            "ifeval": {
                "prompt_level_strict_acc,none": 0.41,
                "prompt_level_strict_acc_stderr,none": 0.02,
            },
            "mmlu": {
                "acc,none": 0.37,
                "acc_stderr,none": 0.01,
            },
        }
    }
    metrics = extract_task_metrics(raw)
    assert extract_ifeval_score(metrics) == pytest.approx(0.41)
    assert extract_MMLU_acc(metrics) == pytest.approx(0.37)


def test_run_skills_eval_writes_json_and_flags(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    out = tmp_path / "skills.json"

    result = run_skills_eval(model_dir, output_path=out, baseline_mmlu_acc=0.40)

    assert result.ifeval_prompt_strict == pytest.approx(0.55)
    assert result.mmlu_acc == pytest.approx(0.30)
    assert result.mmlu_drop is not None
    assert result.mmlu_drop.flagged is True
    assert result.mmlu_drop.drop_pts == pytest.approx(10.0)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mmlu_acc"] == pytest.approx(0.30)
    assert payload["mmlu_drop"]["flagged"] is True


def test_run_skills_eval_default_kwargs_reach_lm_eval(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    """the contract a real P6-0 run depends on; previously bypassed by evaluate_fn."""
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    run_skills_eval(model_dir, output_path=tmp_path / "skills.json")

    assert len(fake_lm_eval.calls) == 1
    kw = fake_lm_eval.calls[0]
    assert kw["model"] == DEFAULT_MODEL_BACKEND == "vllm"
    assert kw["model_args"] == f"pretrained={model_dir}"
    assert kw["tasks"] == ["ifeval", "mmlu"]
    assert kw["batch_size"] == DEFAULT_BATCH_SIZE == "auto"
    # omitted when unset so lm-eval applies its own defaults
    assert "device" not in kw
    assert "limit" not in kw


def test_run_skills_eval_forwards_optional_kwargs(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    run_skills_eval(
        model_dir,
        tasks=["ifeval"],
        model_args_extra="gpu_memory_utilization=0.85,max_model_len=4096",
        batch_size=8,
        device="cuda",
        limit=10,
        output_path=tmp_path / "skills.json",
        num_fewshot=0,  # arbitrary passthrough via **extra_kwargs
    )

    kw = fake_lm_eval.calls[0]
    assert kw["model"] == DEFAULT_MODEL_BACKEND == "vllm"
    assert kw["model_args"] == (
        f"pretrained={model_dir},gpu_memory_utilization=0.85,max_model_len=4096"
    )
    assert kw["tasks"] == ["ifeval"]
    assert kw["batch_size"] == 8
    assert kw["device"] == "cuda"
    assert kw["limit"] == 10
    assert kw["num_fewshot"] == 0


def test_run_skills_eval_requires_tasks(tmp_path: Path) -> None:
    """validation must happen before lm-eval is imported (no fixture on purpose)."""
    with pytest.raises(ValueError, match="tasks"):
        run_skills_eval(tmp_path, tasks=[])

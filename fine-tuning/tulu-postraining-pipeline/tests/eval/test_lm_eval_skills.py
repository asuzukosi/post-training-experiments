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
    assert ok.mmlu_diff == pytest.approx(-4.0)

    bad = flag_MMLU_drop(0.50, 0.44)  # 6 pts
    assert bad.flagged is True
    assert bad.mmlu_diff == pytest.approx(-6.0)

    edge = flag_MMLU_drop(0.50, 0.45)  # exactly 5 pts -> not > 5
    assert edge.flagged is False



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



def test_run_skills_eval_default_kwargs_reach_lm_eval(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    """the contract a real P6-0 run depends on; previously bypassed by evaluate_fn."""
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    run_skills_eval(model_dir, output_path=tmp_path / "skills.json")

    # one call per limit group: ifeval runs whole, mmlu at 25 per subject. a single call
    # cannot express both, since lm-eval's `limit` is global across the task list.
    assert len(fake_lm_eval.calls) == 2
    by_task = {tuple(c["tasks"]): c for c in fake_lm_eval.calls}
    assert set(by_task) == {("ifeval",), ("mmlu",)}

    ifeval, mmlu = by_task[("ifeval",)], by_task[("mmlu",)]
    assert "limit" not in ifeval, "ifeval's 541 questions must not be truncated"
    assert mmlu["limit"] == 25, "25 per subject x 57 = the 1,425 the +/-1.3 figure assumes"

    for kw in fake_lm_eval.calls:
        assert kw["model"] == DEFAULT_MODEL_BACKEND == "vllm"
        # the vllm memory guard ships in every run's model_args, not just tuned calls:
        # without it mmlu OOMs at any --limit. see DEFAULT_VLLM_ARGS.
        model_args = dict(p.split("=", 1) for p in kw["model_args"].split(","))
        assert model_args["pretrained"] == str(model_dir)
        assert model_args["gpu_memory_utilization"] == "0.45"
        assert model_args["max_num_batched_tokens"] == "2048"
        assert kw["batch_size"] == DEFAULT_BATCH_SIZE == "auto"
        # omitted when unset so lm-eval applies its own defaults
        assert "device" not in kw




def test_build_model_args_carries_vllm_defaults() -> None:
    """mmlu OOMs at vllm's defaults; the guard has to be in the args every run gets."""
    from eval.lm_eval_skills import build_model_args

    args = dict(p.split("=", 1) for p in build_model_args("/ckpt").split(","))
    assert args["pretrained"] == "/ckpt"
    assert args["gpu_memory_utilization"] == "0.45"
    assert args["max_num_batched_tokens"] == "2048"
    # vllm rejects a token budget below max_model_len, so the two move together
    assert args["max_model_len"] == "2048"


def test_build_model_args_extra_overrides_by_key() -> None:
    """a caller tuning one knob must replace the default, not append a duplicate."""
    from eval.lm_eval_skills import build_model_args

    out = build_model_args("/ckpt", "gpu_memory_utilization=0.5,dtype=bfloat16")
    assert out.count("gpu_memory_utilization") == 1
    args = dict(p.split("=", 1) for p in out.split(","))
    assert args["gpu_memory_utilization"] == "0.5"
    assert args["max_num_batched_tokens"] == "2048"
    assert args["dtype"] == "bfloat16"


def test_a_hub_id_is_not_turned_into_a_local_path(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    """the bug that killed the first baseline run.

    `resolve_path` glues a relative-looking hub id onto the repo root; the failure then
    surfaces much later, as HFValidationError from inside huggingface_hub, by which point
    it reads like a bad model name rather than a path bug.
    """
    result = run_skills_eval("Qwen/Qwen2.5-1.5B", output_path=tmp_path / "skills.json")

    assert result.model == "Qwen/Qwen2.5-1.5B"
    for call in fake_lm_eval.calls:
        args = dict(p.split("=", 1) for p in call["model_args"].split(","))
        assert args["pretrained"] == "Qwen/Qwen2.5-1.5B"


def test_a_local_checkpoint_is_still_resolved(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    """the other half: a real checkpoint directory must still become an absolute path."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    result = run_skills_eval(ckpt, output_path=tmp_path / "skills_local.json")

    assert result.model == str(ckpt)


def test_chat_template_flag_reaches_lm_eval_and_is_recorded(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    """the protocol has to reach the harness AND be written down.

    a score that does not say whether it was templated cannot be compared to one that
    does — which is exactly how the base-vs-SFT ifeval delta became unreadable.
    """
    result = run_skills_eval(
        "Qwen/Qwen2.5-1.5B",
        output_path=tmp_path / "skills_chat.json",
        apply_chat_template=True,
    )

    assert all(c["apply_chat_template"] is True for c in fake_lm_eval.calls)
    assert result.apply_chat_template is True
    assert json.loads((tmp_path / "skills_chat.json").read_text())["apply_chat_template"]


def test_untemplated_is_the_default_and_is_also_recorded(
    tmp_path: Path, fake_lm_eval: SimpleNamespace
) -> None:
    result = run_skills_eval("Qwen/Qwen2.5-1.5B", output_path=tmp_path / "skills.json")

    assert all(c["apply_chat_template"] is False for c in fake_lm_eval.calls)
    assert result.apply_chat_template is False


def test_templated_run_does_not_overwrite_the_untemplated_one(
    tmp_path: Path, fake_lm_eval: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """both cells of the comparison must survive; the default path is name-derived.

    DEFAULT_METRICS_DIR is redirected because it is the REAL results directory — an
    earlier version of this test wrote fixture values over a genuine baseline run.
    """
    monkeypatch.setattr("eval.lm_eval_skills.DEFAULT_METRICS_DIR", tmp_path / "metrics")

    raw = run_skills_eval("Qwen/Qwen2.5-1.5B")
    chat = run_skills_eval("Qwen/Qwen2.5-1.5B", apply_chat_template=True)

    assert raw.output_path != chat.output_path
    assert Path(raw.output_path).name == "skills_Qwen2.5-1.5B.json"
    assert Path(chat.output_path).name == "skills_Qwen2.5-1.5B_chat.json"
    assert Path(raw.output_path).parent == tmp_path / "metrics"

"""unit tests for lm-eval skills wrapper (no harness / no gpu)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.eval.lm_eval_skills import (
    extract_ifeval_score,
    extract_MMLU_acc,
    extract_task_metrics,
    flag_MMLU_drop,
    run_skills_eval,
)


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


def test_run_skills_eval_writes_json_and_flags(tmp_path: Path) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    out = tmp_path / "skills.json"

    def fake_evaluate(**kwargs):
        assert kwargs["tasks"] == ["ifeval", "mmlu"]
        assert "pretrained=" in kwargs["model_args"]
        return {
            "results": {
                "ifeval": {"prompt_level_strict_acc,none": 0.55},
                "mmlu": {"acc,none": 0.30},
            }
        }

    result = run_skills_eval(
        model_dir,
        output_path=out,
        baseline_mmlu_acc=0.40,
        evaluate_fn=fake_evaluate,
    )
    assert result.ifeval_prompt_strict == pytest.approx(0.55)
    assert result.mmlu_acc == pytest.approx(0.30)
    assert result.mmlu_drop is not None
    assert result.mmlu_drop.flagged is True
    assert result.mmlu_drop.drop_pts == pytest.approx(10.0)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mmlu_acc"] == pytest.approx(0.30)
    assert payload["mmlu_drop"]["flagged"] is True


def test_run_skills_eval_requires_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tasks"):
        run_skills_eval(
            tmp_path,
            tasks=[],
            evaluate_fn=lambda **_: {"results": {}},
        )

"""lm-eval harness wrapper for ifeval + mmlu skills guardrails.

mmlu drop >5 percentage points vs a baseline is a broken-run flag (spec).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prepare.paths import ROOT, resolve_path

DEFAULT_TASKS = ("ifeval", "mmlu") # tasks to evaluate
DEFAULT_MODEL_BACKEND = "vllm"
DEFAULT_BATCH_SIZE = "auto" # batch size for lm-eval

DEFAULT_VLLM_ARGS = (
    "gpu_memory_utilization=0.45,max_num_batched_tokens=2048,max_model_len=2048" # vllm args
)
DEFAULT_MAX_MMLU_DROP = 5.0 # mmlu drop threshold in percentage points
DEFAULT_METRICS_DIR = ROOT / "results" / "metrics" # where the metrics are written to

# metric key preferences inside lm-eval results[task]
_MMLU_METRIC_KEYS = (
    "acc,none",
    "acc",
    "exact_match,none",
    "exact_match",
)
_IFEVAL_METRIC_KEYS = (
    "prompt_level_strict_acc,none",
    "prompt_level_strict_acc",
    "inst_level_strict_acc,none",
    "inst_level_strict_acc",
)


@dataclass
class MMLUDropFlag:
    """mmlu against a baseline, flagged when it has fallen too far."""

    baseline_acc: float
    current_acc: float
    # points against the baseline: POSITIVE is a lift, negative is a drop. same
    # convention as the training tripwire and the attribution deltas.
    mmlu_diff: float
    max_drop: float = DEFAULT_MAX_MMLU_DROP
    flagged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillsEvalResult:
    """summary of an ifeval/mmlu lm-eval run."""

    model: str
    tasks: list[str]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    ifeval_prompt_strict: float | None = None
    mmlu_acc: float | None = None
    mmlu_drop: MMLUDropFlag | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def flag_MMLU_drop(
    baseline_acc: float,
    current_acc: float,
    *,
    max_drop: float = DEFAULT_MAX_MMLU_DROP,
) -> MMLUDropFlag:
    """flag when mmlu has fallen more than `max_drop` points below the baseline.

    acc values are fractions in [0, 1]; the difference is reported in points.
    """
    if not 0.0 <= baseline_acc <= 1.0:
        raise ValueError(f"baseline_acc must be in [0, 1], got {baseline_acc}")
    if not 0.0 <= current_acc <= 1.0:
        raise ValueError(f"current_acc must be in [0, 1], got {current_acc}")
    if max_drop < 0:
        raise ValueError(f"max_drop must be >= 0, got {max_drop}")

    mmlu_diff = (current_acc - baseline_acc) * 100.0
    return MMLUDropFlag(
        baseline_acc=baseline_acc,
        current_acc=current_acc,
        mmlu_diff=mmlu_diff,
        max_drop=max_drop,
        flagged=mmlu_diff < -max_drop,
    )


def _pick_metric(task_metrics: Mapping[str, Any], candidates: Sequence[str]) -> float | None:
    for key in candidates:
        if key in task_metrics:
            try:
                return float(task_metrics[key])
            except (TypeError, ValueError):
                continue
    # fallback: first numeric value whose name starts with acc / exact / prompt_level
    for key, val in task_metrics.items():
        if key.endswith("_stderr") or "stderr" in key:
            continue
        low = key.lower()
        if low.startswith(("acc", "exact", "prompt_level", "inst_level")):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def extract_task_metrics(raw_results: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """flatten lm-eval `results` block to {task: {metric: float}} for numeric fields."""
    block = raw_results.get("results", raw_results)
    if not isinstance(block, Mapping):
        raise ValueError("lm-eval results missing 'results' mapping")
    out: dict[str, dict[str, float]] = {}
    for task, metrics in block.items():
        if not isinstance(metrics, Mapping):
            continue
        cleaned: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                cleaned[str(k)] = float(v)
        if cleaned:
            out[str(task)] = cleaned
    return out


def extract_ifeval_score(metrics: Mapping[str, Mapping[str, float]]) -> float | None:
    for name in ("ifeval",):
        if name in metrics:
            return _pick_metric(metrics[name], _IFEVAL_METRIC_KEYS)
    for task, vals in metrics.items():
        if "ifeval" in task.lower():
            score = _pick_metric(vals, _IFEVAL_METRIC_KEYS)
            if score is not None:
                return score
    return None


def extract_MMLU_acc(metrics: Mapping[str, Mapping[str, float]]) -> float | None:
    # prefer aggregate "mmlu" over subject slices
    if "mmlu" in metrics:
        return _pick_metric(metrics["mmlu"], _MMLU_METRIC_KEYS)
    for task, vals in metrics.items():
        if task.lower() == "mmlu" or task.lower().startswith("mmlu_"):
            score = _pick_metric(vals, _MMLU_METRIC_KEYS)
            if score is not None:
                return score
    return None


def build_model_args(model_path: str | Path, extra: str = "") -> str:
    """`pretrained=...` plus the vllm defaults, with `extra` overriding by key.

    lm-eval parses this as comma-separated key=value, so merging by key rather than
    concatenating means a caller passing its own gpu_memory_utilization replaces the
    default instead of appending a duplicate whose precedence is undefined.
    """
    args: dict[str, str] = {"pretrained": str(model_path)}
    for chunk in (DEFAULT_VLLM_ARGS, extra):
        for part in str(chunk).split(","):
            part = part.strip()
            if not part:
                continue
            key, _, value = part.partition("=")
            args[key.strip()] = value.strip()
    model_args = ",".join(f"{k}={v}" for k, v in args.items())
    print(f"model_args: {model_args}")
    return model_args


def _default_simple_evaluate(
    *,
    model: str,
    model_args: str,
    tasks: list[str],
    batch_size: str | int,
    device: str | None,
    limit: float | int | None,
    extra_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    from lm_eval import simple_evaluate

    kwargs: dict[str, Any] = {
        "model": model,
        "model_args": model_args,
        "tasks": tasks,
        "batch_size": batch_size,
    }
    if device is not None:
        kwargs["device"] = device
    if limit is not None:
        kwargs["limit"] = limit
    kwargs.update(dict(extra_kwargs))
    return simple_evaluate(**kwargs)


def run_skills_eval(
    model_path: str | Path,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    model_args_extra: str = "",
    batch_size: str | int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    limit: float | int | None = None,
    output_path: str | Path | None = None,
    baseline_mmlu_acc: float | None = None,
    max_mmlu_drop: float = DEFAULT_MAX_MMLU_DROP,
    **extra_kwargs: Any,
) -> SkillsEvalResult:
    """run lm-eval ifeval/mmlu (or custom task list); write metrics json.

    there is one path: `_default_simple_evaluate` -> `lm_eval.simple_evaluate`.
    tests patch `sys.modules["lm_eval"]` instead of injecting a callable, so the
    kwargs assembly is exercised rather than bypassed. `baseline_mmlu_acc`
    enables the >5pt broken-run flag when mmlu is among the tasks.
    """
    path = resolve_path(model_path)
    task_list = [str(t) for t in tasks]
    if not task_list:
        raise ValueError("tasks must be non-empty")

    model_args = build_model_args(path, model_args_extra)

    print(
        f"lm-eval: backend={DEFAULT_MODEL_BACKEND} model={path} "
        f"tasks={task_list} batch_size={batch_size}"
    )

    raw = _default_simple_evaluate(
        model=DEFAULT_MODEL_BACKEND,
        model_args=model_args,
        tasks=task_list,
        batch_size=batch_size,
        device=device,
        limit=limit,
        extra_kwargs=extra_kwargs,
    )

    metrics = extract_task_metrics(raw)
    ifeval_score = extract_ifeval_score(metrics)
    mmlu_acc = extract_MMLU_acc(metrics)

    mmlu_drop: MMLUDropFlag | None = None
    if baseline_mmlu_acc is not None and mmlu_acc is not None:
        mmlu_drop = flag_MMLU_drop(
            baseline_mmlu_acc,
            mmlu_acc,
            max_drop=max_mmlu_drop,
        )
        if mmlu_drop.flagged:
            print(
                f"mmlu drop flagged: baseline={baseline_mmlu_acc:.4f} "
                f"current={mmlu_acc:.4f} mmlu_diff={mmlu_drop.mmlu_diff:+.2f} "
                f"past the {max_mmlu_drop:g} point limit"
            )
        else:
            print(
                f"mmlu ok: mmlu_diff={mmlu_drop.mmlu_diff:+.2f} "
                f"(limit {max_mmlu_drop:g})"
            )

    out = (
        resolve_path(output_path)
        if output_path is not None
        else DEFAULT_METRICS_DIR / f"skills_{path.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    result = SkillsEvalResult(
        model=str(path),
        tasks=task_list,
        metrics=metrics,
        ifeval_prompt_strict=ifeval_score,
        mmlu_acc=mmlu_acc,
        mmlu_drop=mmlu_drop,
        output_path=str(out),
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
        f.write("\n")
    print(
        f"lm-eval done: ifeval={ifeval_score} mmlu={mmlu_acc} wrote={out}"
    )
    return result

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

DEFAULT_TASKS = ("ifeval", "mmlu")
# always vllm. switching to hf mid-campaign injects ~0.1-0.5pt noise into the
# mmlu deltas the attribution table reads against DEFAULT_MMLU_DROP_PTS.
DEFAULT_MODEL_BACKEND = "vllm"
DEFAULT_BATCH_SIZE = "auto"
# vllm reserves gpu_memory_utilization of the card up front for weights + kv cache,
# but mmlu is scored by loglikelihood, which needs logits at EVERY prompt position
# rather than just the last one, and that tensor is allocated outside the reservation:
# max_num_batched_tokens x vocab x 4 bytes. at vllm's defaults on a 24gb card that is
# 16384 x 151936 x 4 = 9.27 GiB against ~3 GiB free, so every mmlu run OOMs — at
# limit=5 exactly as hard as limit=25, since the allocation does not depend on the
# document count. capping the token budget cuts the tensor to 4096 x 151936 x 4 =
# 2.49 GiB. the vocab term does NOT shrink with model size, so this gets tighter, not
# looser, at 1.5B. generative tasks (ifeval) are unaffected: they need logits for one
# position per sequence, ~156 MB.
#
# max_model_len has to come down with it: vllm refuses a token budget smaller than the
# model's context (qwen2.5 declares 32768) rather than silently truncating. 4096 is well
# clear of what these tasks need — mmlu 5-shot runs ~900 tokens and ifeval prompts are
# shorter — and it shrinks the kv cache too, which buys back the utilisation we gave up.
#
# the token budget is the lever, not the reservation. every tensor in the scoring path
# is (tokens x vocab), so halving the budget halves ALL of them at once, whereas cutting
# gpu_memory_utilization buys a fixed amount of headroom against a peak that kept turning
# out bigger. walking the failures down sampler.py at 4096 tokens: 0.80 died on the fp32
# log_softmax (2.11 GiB), 0.65 got past it and died on the gather (2.11 GiB), 0.55 got
# past that and died in _get_ranks on `result.sum(1)` — 4.21 GiB, because the bool
# comparison promotes to int64 at 8 bytes per element. peak transient is ~9 GiB, and
# there may be more of it that no run has reached yet.
#
# at 2048 tokens every one of those halves (~4.5 GiB peak), and 0.45 leaves ~13 GiB free
# against it. the margin no longer depends on having found the true peak. what remains
# reserved still covers weights (0.93 GiB at 0.5B, ~3 at 1.5B), activations, and a kv
# cache far larger than 2048-token evals can use.
#
# max_model_len tracks the budget because vllm rejects a budget below the context length
# rather than truncating. 2048 clears these tasks: mmlu docs run a few hundred tokens and
# ifeval is short prompts plus a bounded generation — it already passed at 4096.
DEFAULT_VLLM_ARGS = (
    "gpu_memory_utilization=0.45,max_num_batched_tokens=2048,max_model_len=2048"
)
DEFAULT_MMLU_DROP_PTS = 5.0
DEFAULT_METRICS_DIR = ROOT / "results" / "metrics"

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
    """compare current mmlu acc to a baseline; flag if drop exceeds threshold pts."""

    baseline_acc: float
    current_acc: float
    drop_pts: float
    threshold_pts: float = DEFAULT_MMLU_DROP_PTS
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
    threshold_pts: float = DEFAULT_MMLU_DROP_PTS,
) -> MMLUDropFlag:
    """flag when current mmlu is more than `threshold_pts` points below baseline.

    acc values are fractions in [0, 1]; drop is reported in percentage points.
    """
    if not 0.0 <= baseline_acc <= 1.0:
        raise ValueError(f"baseline_acc must be in [0, 1], got {baseline_acc}")
    if not 0.0 <= current_acc <= 1.0:
        raise ValueError(f"current_acc must be in [0, 1], got {current_acc}")
    if threshold_pts < 0:
        raise ValueError(f"threshold_pts must be >= 0, got {threshold_pts}")

    drop_pts = (baseline_acc - current_acc) * 100.0
    flagged = drop_pts > threshold_pts
    return MMLUDropFlag(
        baseline_acc=baseline_acc,
        current_acc=current_acc,
        drop_pts=drop_pts,
        threshold_pts=threshold_pts,
        flagged=flagged,
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
    return ",".join(f"{k}={v}" for k, v in args.items())


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
    mmlu_drop_threshold_pts: float = DEFAULT_MMLU_DROP_PTS,
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
            threshold_pts=mmlu_drop_threshold_pts,
        )
        if mmlu_drop.flagged:
            print(
                f"mmlu drop flagged: baseline={baseline_mmlu_acc:.4f} "
                f"current={mmlu_acc:.4f} drop_pts={mmlu_drop.drop_pts:.2f} "
                f"> {mmlu_drop_threshold_pts:g}"
            )
        else:
            print(
                f"mmlu drop ok: drop_pts={mmlu_drop.drop_pts:.2f} "
                f"(threshold={mmlu_drop_threshold_pts:g})"
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

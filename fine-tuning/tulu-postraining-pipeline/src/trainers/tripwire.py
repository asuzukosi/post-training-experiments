"""mid-training mmlu tripwire: stop a run that has collapsed, before it finishes.

runs lm-eval against the model ALREADY IN MEMORY via the hf backend. there is no second
copy of the model and no engine handover — vllm would reserve ~90% of the card and the
training model plus its optimizer state are already on it.

this is deliberately not the evaluation path. `src/eval/` owns every number that reaches
the report and always goes through vllm; this only answers "has the run collapsed". the
two use different backends, so their numbers are close but not interchangeable — never
put a tripwire reading in the attribution table.

a small sample is the point. catching a 20-point collapse needs nothing like the
precision of the final measurement, and the run is paying for every question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from eval.io import append_jsonl

DEFAULT_EVALS = 4  # mid-run checks, evenly spaced, whatever the run length
DEFAULT_QUESTIONS_PER_SUBJECT = 5  # mmlu has 57 subjects -> 285 questions
DEFAULT_MAX_DROP = 5.0  # points of mmlu; the run stops if it falls further than this
DEFAULT_BATCH_SIZE = 4  # loglikelihood materialises logits per position; keep it small


def measure_mmlu(model: Any, tokenizer: Any, *, limit: int, batch_size: int) -> float | None:
    """mmlu accuracy for a live model, via lm-eval's hf backend."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    was_training = bool(model.training)
    model.eval()
    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        raw = simple_evaluate(model=lm, tasks=["mmlu"], limit=limit)
    finally:
        if was_training:
            model.train()

    block = (raw or {}).get("results", {}).get("mmlu", {})
    for key in ("acc,none", "acc"):
        if key in block:
            return float(block[key])
    return None


def eval_steps(max_steps: int, evals: int = DEFAULT_EVALS) -> list[int]:
    """evenly spaced steps for `evals` mid-run checks.

    a fixed FRACTION of the run, not a fixed step count: sft runs ~1,500 optimizer steps
    and ppo ~46, so any single step interval would over-sample one and skip the other
    entirely.

    the end of the run is excluded — that measurement is the final evaluation's job, and
    it is done properly on the full question set through vllm.
    """
    if max_steps < 2 or evals < 1:
        return []
    interval = max(1, max_steps // (evals + 1))
    steps = [interval * i for i in range(1, evals + 1)]
    return sorted({s for s in steps if 0 < s < max_steps})


@dataclass
class MMLUTripwire(TrainerCallback):
    """abort a training run whose mmlu has collapsed.

    subclasses TrainerCallback for its no-op defaults: the handler dispatches with
    `getattr(callback, event)` and fires fifteen events, so a duck-typed object raises
    AttributeError on the first one it does not implement.

    it aborts rather than warns on purpose. a tripwire that only logs is a slower way of
    finding out at the end, by which point the gpu time is already spent. step
    checkpoints are already on disk, so an aborted run is inspectable and resumable.
    """

    tokenizer: Any
    limit: int = DEFAULT_QUESTIONS_PER_SUBJECT
    batch_size: int = DEFAULT_BATCH_SIZE
    max_drop: float = DEFAULT_MAX_DROP
    evals: int = DEFAULT_EVALS
    baseline: float | None = None  # pre-training mmlu; the first reading if unset
    # where the trajectory is written. every reading is flushed immediately, so an
    # aborted run still leaves the evidence of why it stopped.
    output_dir: str | Path | None = None
    readings: list[dict[str, Any]] = field(default_factory=list)
    _steps: list[int] = field(default_factory=list)

    @property
    def readings_path(self) -> Path | None:
        return None if self.output_dir is None else Path(self.output_dir) / "tripwire_mmlu.jsonl"

    def _record(self, step: int, mmlu: float, mmlu_diff: float | None) -> None:
        """append one reading, and mirror it to w&b if a run is already open.

        jsonl rather than one rewritten json: appending cannot truncate what is already
        there, which matters because the run this file explains is the one that dies.
        each line carries the baseline and threshold so it stands alone.
        """
        record = {
            "step": step,
            "mmlu": mmlu,
            # points against the baseline: POSITIVE is a lift, negative is a drop
            "mmlu_diff": mmlu_diff,
            "baseline_mmlu": self.baseline,
            "max_drop": self.max_drop,
            "questions_per_subject": self.limit,
            "backend": "hf",  # not the vllm number; never mix these into the table
        }
        self.readings.append(record)
        if self.readings_path is not None:
            append_jsonl(self.readings_path, record)
        self._log_to_wandb(step, mmlu, mmlu_diff)

    @staticmethod
    def _log_to_wandb(step: int, mmlu: float, mmlu_diff: float | None) -> None:
        """piggyback on the trainer's run if one is open; never start one."""
        try:
            import wandb
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return
        payload: dict[str, Any] = {"tripwire/mmlu": mmlu}
        if mmlu_diff is not None:
            payload["tripwire/mmlu_diff"] = mmlu_diff
        wandb.log(payload, step=step)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._steps = eval_steps(int(state.max_steps or 0), self.evals)
        print(f"mmlu tripwire: checking at steps {self._steps or '(run too short)'}")
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step not in self._steps:
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        acc = measure_mmlu(
            model, self.tokenizer, limit=self.limit, batch_size=self.batch_size
        )
        if acc is None:
            print(f"mmlu tripwire: step {step} returned no score; skipping")
            return control

        if self.baseline is None:
            self.baseline = acc
            self._record(step, acc, None)
            print(f"mmlu tripwire: step {step} mmlu={acc:.4f} (baseline)")
            return control

        # positive is a lift, negative is a drop — the same sign convention the
        # attribution deltas use, so the two never have to be mentally inverted.
        mmlu_diff = (acc - self.baseline) * 100.0
        self._record(step, acc, mmlu_diff)
        print(
            f"mmlu tripwire: step {step} mmlu={acc:.4f} "
            f"baseline={self.baseline:.4f} diff={mmlu_diff:+.2f} points"
        )
        if mmlu_diff < -self.max_drop:
            raise RuntimeError(
                f"mmlu tripwire: {mmlu_diff:+.2f} points by step {step} "
                f"({self.baseline:.4f} -> {acc:.4f}), past the {self.max_drop:g} "
                "point drop limit. the run is stopping; the last checkpoint is on disk."
            )
        return control


def attach_mmlu_tripwire(trainer: Any, cfg: dict[str, Any], tokenizer: Any) -> None:
    """add the mmlu tripwire to `trainer`, if `tripwire_evals` asks for one.

    a no-op at 0 or unset, so a run never pays for mid-training evaluation by accident.
    """
    evals = int(cfg.get("tripwire_evals", 0))
    if evals < 1:
        return
    trainer.add_callback(
        MMLUTripwire(
            tokenizer=tokenizer,
            evals=evals,
            limit=int(cfg.get("tripwire_questions", DEFAULT_QUESTIONS_PER_SUBJECT)),
            batch_size=int(cfg.get("tripwire_batch_size", DEFAULT_BATCH_SIZE)),
            max_drop=float(cfg.get("tripwire_max_drop", DEFAULT_MAX_DROP)),
            output_dir=getattr(getattr(trainer, "args", None), "output_dir", None),
            baseline=(
                None
                if cfg.get("tripwire_baseline_mmlu") is None
                else float(cfg["tripwire_baseline_mmlu"])
            ),
        )
    )

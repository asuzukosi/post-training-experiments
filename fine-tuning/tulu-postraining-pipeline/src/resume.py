"""shared resume helpers for volume-backed training.

- auto-detect latest `checkpoint-*` under the run output dir
- persist a fixed wandb run id so remount continues the same run with resume=allow


ckpt = resolve_resume_from_checkpoint(output_dir, resume=True)
with wandb_run(use_wandb=True, output_dir=output_dir, project="tulu-pt", name=run_name):
    trainer.train(resume_from_checkpoint=ckpt)
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


class WandbResumeMode(str, Enum):
    """wandb.init(resume=...) modes.

    allow  — resume if id exists, else create new run with that id
    must   — resume if id exists, else error
    never  — error if id exists, else create new run with that id
    auto   — resume crashed run on same machine/dir if possible, else new run
    """

    ALLOW = "allow"
    MUST = "must"
    NEVER = "never"
    AUTO = "auto"


DEFAULT_WANDB_RESUME = WandbResumeMode.ALLOW


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    """return path to the highest-step `checkpoint-*` under `output_dir`, or none."""
    root = Path(output_dir)
    if not root.is_dir():
        return None

    best_step = -1
    best_path: Path | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_DIR_RE.match(child.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = child
    return str(best_path) if best_path is not None else None


def resolve_resume_from_checkpoint(
    output_dir: str | Path,
    resume: bool | str = True,
) -> bool | str:
    """value for `trainer.train(resume_from_checkpoint=...)`.

    - `resume=False` -> False (fresh run)
    - `resume="/path/to/checkpoint-N"` -> that path (must exist)
    - `resume=True` -> latest checkpoint path if present, else False
    """
    if resume is False:
        return False
    if isinstance(resume, str):
        path = Path(resume)
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return str(path)

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        print(f"no checkpoint under {output_dir}; starting fresh")
        return False
    print(f"resuming from {latest}")
    return latest


def wandb_run_id_path(output_dir: str | Path) -> Path:
    """path where the fixed wandb run id is stored for this training run dir."""
    return Path(output_dir) / WANDB_RUN_ID_FILENAME


def get_or_create_wandb_run_id(
    output_dir: str | Path,
    run_id: str | None = None,
) -> str:
    """load persisted wandb run id, or create + write one.

    keeps metrics on the same wandb run across pod remounts.
    output_dir: the directory where the wandb run id is stored e.g "/workspace/checkpoints/qwen2.5-1.5b_sft_20260812T1200Z"
    """
    path = wandb_run_id_path(output_dir)
    if path.is_file():
        existing = path.read_text().strip()
        if existing:
            return existing

    if run_id and run_id.strip():
        rid = run_id.strip()
    else:
        import wandb

        # generate a new wandb run id for new run
        rid = wandb.util.generate_id()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rid + "\n")
    print(f"wandb run id: {rid} ({path})")
    return rid


def parse_wandb_resume_mode(resume: str | WandbResumeMode) -> WandbResumeMode:
    """coerce config/cli strings to WandbResumeMode."""
    if isinstance(resume, WandbResumeMode):
        return resume
    try:
        return WandbResumeMode(str(resume).strip().lower())
    except ValueError as e:
        allowed = ", ".join(m.value for m in WandbResumeMode)
        raise ValueError(
            f"unknown wandb resume mode {resume!r}; expected one of: {allowed}"
        ) from e


def wandb_resume_kwargs(
    output_dir: str | Path,
    *,
    project: str,
    name: str,
    resume: str | WandbResumeMode = DEFAULT_WANDB_RESUME,
    entity: str | None = None,
    run_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """kwargs for `wandb.init(...)` with fixed id + resume mode.

    example:
    {
        "project": "qwen2.5-1.5b_sft",
        "name": "qwen2.5-1.5b_sft_20260812T1200Z",
        "id": "qwen2.5-1.5b_sft_20260812T1200Z",
        "resume": "allow",
    }
    wandb_kwargs = wandb_resume_kwargs(output_dir, project="qwen2.5-1.5b_sft", name="qwen2.5-1.5b_sft_20260812T1200Z", resume="allow")
    wandb.init(**wandb_kwargs)
    """
    if not project:
        raise ValueError("project must be non-empty")
    if not name:
        raise ValueError("name must be non-empty")

    mode = parse_wandb_resume_mode(resume)
    kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "id": get_or_create_wandb_run_id(output_dir, run_id=run_id),
        "resume": mode.value,
    }
    if entity:
        kwargs["entity"] = entity
    kwargs.update(extra)
    return kwargs


def cfg_use_wandb(cfg: Mapping[str, Any], default: bool = True) -> bool:
    """read the use_wandb config flag (default on)."""
    if "use_wandb" not in cfg:
        return default
    return bool(cfg["use_wandb"])


def trainer_report_to(use_wandb: bool) -> str:
    """hf trainer report_to derived from use_wandb (wandb-only stack)."""
    return "wandb" if use_wandb else "none"


@contextmanager
def wandb_run(
    *,
    use_wandb: bool,
    output_dir: str | Path,
    project: str,
    name: str,
    resume: str | WandbResumeMode = DEFAULT_WANDB_RESUME,
    config: Mapping[str, Any] | None = None,
    entity: str | None = None,
) -> Iterator[Any]:
    """init/finish wandb when `use_wandb`; otherwise a no-op that yields none.

    example:
      with wandb_run(use_wandb=True, output_dir=out, project=..., name=...) as run:
          trainer.train(...)
    """
    if not use_wandb:
        yield None
        return

    import wandb

    wb_kwargs = wandb_resume_kwargs(
        output_dir,
        project=project,
        name=name,
        resume=resume,
        entity=entity,
    )
    run = wandb.init(**wb_kwargs)
    if config:
        wandb.config.update(dict(config), allow_val_change=True)
    try:
        yield run
    finally:
        wandb.finish()

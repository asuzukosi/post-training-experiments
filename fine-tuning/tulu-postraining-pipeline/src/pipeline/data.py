"""dataset prep helpers. p1-1: run naming + checkpoint paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# default on-volume layout from the spec
DEFAULT_CHECKPOINTS_ROOT = Path("results/checkpoints")


def make_run_name(
    base_name: str,
    task: str,
    when: datetime | None = None,
) -> str:
    """build `<base_name>_<task>_<YYYYMMDDThhmmZ>` (utc).

    examples:
      make_run_name("qwen2.5-1.5b", "sft") -> "qwen2.5-1.5b_sft_20260812T1200Z"
      make_run_name("qwen2.5-1.5b", "dpo-b0.05") -> "qwen2.5-1.5b_dpo-b0.05_20260812T1200Z"
    """
    when = when or datetime.now(timezone.utc)
    stamp = when.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{base_name}_{task}_{stamp}"


def checkpoint_dir(
    run_name: str,
    checkpoints_root: str | Path = DEFAULT_CHECKPOINTS_ROOT,
) -> Path:
    """return `results/checkpoints/<run_name>` (or under `checkpoints_root`)."""
    return Path(checkpoints_root) / run_name


def make_checkpoint_dir(
    base_name: str,
    task: str,
    when: datetime | None = None,
    checkpoints_root: str | Path = DEFAULT_CHECKPOINTS_ROOT,
) -> Path:
    """convenience: run name + checkpoint path in one call."""
    return checkpoint_dir(make_run_name(base_name, task, when=when), checkpoints_root)



def create_checkpoint_dir(
    base_name: str,
    task: str,
    when: datetime | None = None,
    checkpoints_root: str | Path = DEFAULT_CHECKPOINTS_ROOT,
) -> Path:
    """create checkpoint directory and return path."""
    dir = make_checkpoint_dir(base_name, task, when=when, checkpoints_root=checkpoints_root)
    dir.mkdir(parents=True, exist_ok=True)
    return dir
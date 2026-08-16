"""project root and config path helpers for prepare stages."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def resolve_path(path: str | Path) -> Path:
    """resolve relative paths against the repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def model_ref(raw: str | Path) -> str:
    """a model reference: local checkpoint paths resolved, hub ids left alone.

    `resolve_path` cannot be used for models. a hub id like `Qwen/Qwen2.5-1.5B` is
    relative-looking, so it would be glued onto the repo root and then fail repo-id
    validation deep inside huggingface_hub — which is what happened to the first
    baseline eval. anything that names a model has to go through here.
    """
    s = str(raw)
    p = Path(s)
    if p.is_absolute():
        return str(p)
    # project-relative artifact
    if s.startswith(("results/", "data/", "./")) or (ROOT / s).exists():
        return str(resolve_path(s))
    return s  # e.g. Qwen/Qwen2.5-1.5B

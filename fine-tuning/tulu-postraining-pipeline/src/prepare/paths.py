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

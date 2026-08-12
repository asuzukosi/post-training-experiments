"""shared json / path helpers for analysis outputs."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipeline.prepare.paths import ROOT, resolve_path

DEFAULT_METRICS_DIR = ROOT / "results" / "metrics"
DEFAULT_PLOTS_DIR = ROOT / "results" / "plots"


def load_json_mapping(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """load a dict from a mapping or a json file path."""
    if isinstance(source, Mapping):
        return dict(source)
    path = resolve_path(source)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"expected json object at {path}")
    return dict(data)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """write pretty json; creates parent dirs."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
        f.write("\n")
    return out


def resolve_output_path(path: str | Path | None, *, default: Path) -> Path:
    """use explicit path or default under results/; ensure parent exists."""
    out = default if path is None else resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

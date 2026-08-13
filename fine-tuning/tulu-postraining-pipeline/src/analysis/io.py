"""shared json / path helpers for analysis outputs."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prepare.paths import ROOT, resolve_path

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


def parse_stage_pairs(items: Sequence[str]) -> dict[str, str]:
    """parse repeatable STAGE=PATH cli values."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected STAGE=PATH, got {item!r}")
        stage, path = item.split("=", 1)
        stage = stage.strip()
        path = path.strip()
        if not stage or not path:
            raise ValueError(f"expected STAGE=PATH, got {item!r}")
        out[stage] = path
    return out


def merge_stage_map(
    json_path: Path | None,
    pairs: Sequence[str],
) -> dict[str, str]:
    """json map plus STAGE=PATH overrides."""
    out: dict[str, str] = {}
    if json_path is not None:
        payload = load_json_mapping(json_path)
        for stage, path in payload.items():
            out[str(stage)] = str(path)
    out.update(parse_stage_pairs(pairs))
    return out


def load_json_list(path: str | Path) -> list[Any]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected json list at {resolved}")
    return data

"""jsonl helpers for incremental eval artifacts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prepare.paths import resolve_path

ID_KEY = "id"
PROMPT_KEY = "prompt"


def load_completed_ids(output_path: str | Path) -> set[str]:
    """return prompt ids already written to a jsonl file (empty if missing)."""
    path = resolve_path(output_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid jsonl at {path}:{line_no}") from e
            if ID_KEY not in row:
                raise ValueError(f"missing {ID_KEY!r} at {path}:{line_no}")
            done.add(str(row[ID_KEY]))
    return done


def append_jsonl(output_path: str | Path, record: Mapping[str, Any]) -> None:
    """append one json object as a line; creates parent dirs; flushes immediately."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """load all non-empty jsonl rows; missing file -> []."""
    p = resolve_path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid jsonl at {p}:{line_no}") from e
            if not isinstance(row, dict):
                raise ValueError(f"jsonl row must be an object at {p}:{line_no}")
            rows.append(row)
    return rows

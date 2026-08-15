"""jsonl helpers for incremental eval artifacts."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from prepare.paths import resolve_path

ID_KEY = "id"
PROMPT_KEY = "prompt"


def repair_torn_tail(output_path: str | Path) -> bool:
    """drop a half-written final line; return whether one was found.

    a process killed mid-append leaves its last line without a terminating newline.
    that line is the only one a crash can tear — every earlier line was followed by a
    completed write — so it is safe to drop, while a bad line anywhere else is real
    corruption and must still raise.

    dropping it on read alone would not be enough: the next append runs onto the same
    unterminated line, which merges two records into one and moves the damage into the
    middle of the file, where it is unrecoverable rather than merely truncated.
    """
    path = resolve_path(output_path)
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    keep = data.rfind(b"\n") + 1  # 0 when the file is a single torn line
    print(f"repairing torn final line in {path}: dropping {len(data) - keep} bytes")
    with path.open("rb+") as f:
        f.truncate(keep)
    return True


def _iter_rows(path: Path) -> Iterator[tuple[int, Any]]:
    """yield (line_no, row), skipping a torn final line.

    every reader here is eventually handed a file written by a process that may have
    been killed, so an unterminated last line is an expected artifact rather than
    corruption. it is also the ONLY line a crash can tear — each earlier line was
    followed by a completed write — so a bad line anywhere else still raises.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    has_torn_tail = bool(text) and not text.endswith("\n")
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            if has_torn_tail and line_no == len(lines):
                print(f"ignoring torn final line at {path}:{line_no} (killed writer)")
                return
            raise ValueError(f"invalid jsonl at {path}:{line_no}") from e
        yield line_no, row


def load_completed_ids(output_path: str | Path) -> set[str]:
    """return prompt ids already written to a jsonl file (empty if missing)."""
    path = resolve_path(output_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for line_no, row in _iter_rows(path):
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
    for line_no, row in _iter_rows(p):
        if not isinstance(row, dict):
            raise ValueError(f"jsonl row must be an object at {p}:{line_no}")
        rows.append(row)
    return rows

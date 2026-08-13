"""group n-sample generation rows by prompt_id."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eval.io import ID_KEY, PROMPT_KEY

PROMPT_ID_KEY = "prompt_id"
SAMPLE_IDX_KEY = "sample_idx"
COMPLETION_KEY = "completion"


def _require_candidate(row: Mapping[str, Any], *, line_no: int | None = None) -> dict[str, Any]:
    loc = f" at line {line_no}" if line_no is not None else ""
    if PROMPT_ID_KEY not in row:
        raise ValueError(f"generation row missing {PROMPT_ID_KEY!r}{loc}: {row!r}")
    if PROMPT_KEY not in row:
        raise ValueError(f"generation row missing {PROMPT_KEY!r}{loc}: {row!r}")
    if COMPLETION_KEY not in row:
        raise ValueError(f"generation row missing {COMPLETION_KEY!r}{loc}: {row!r}")
    prompt = row[PROMPT_KEY]
    completion = row[COMPLETION_KEY]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"generation row {PROMPT_KEY!r} must be a non-empty str{loc}")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError(
            f"generation row {COMPLETION_KEY!r} must be a non-empty str{loc}"
        )
    prompt_id = str(row[PROMPT_ID_KEY])
    sample_idx = row.get(SAMPLE_IDX_KEY)
    item_id = str(row[ID_KEY]) if ID_KEY in row else f"{prompt_id}__{sample_idx}"
    out = {
        ID_KEY: item_id,
        PROMPT_ID_KEY: prompt_id,
        PROMPT_KEY: prompt,
        COMPLETION_KEY: completion,
        SAMPLE_IDX_KEY: sample_idx,
    }
    for key, value in row.items():
        if key not in out:
            out[key] = value
    return out


def group_candidates(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """group generation rows by prompt_id; assign sample_idx if missing."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for i, row in enumerate(records, start=1):
        cand = _require_candidate(row, line_no=i)
        grouped.setdefault(cand[PROMPT_ID_KEY], []).append(cand)
    out: dict[str, list[dict[str, Any]]] = {}
    for prompt_id, rows in grouped.items():
        prompts = {r[PROMPT_KEY] for r in rows}
        if len(prompts) != 1:
            raise ValueError(
                f"prompt_id {prompt_id!r} has mismatched prompt texts"
            )
        ordered = sorted(rows, key=lambda r: (r[ID_KEY],))
        assigned: list[dict[str, Any]] = []
        used: set[int] = set()
        for i, row in enumerate(ordered):
            raw_idx = row[SAMPLE_IDX_KEY]
            idx = i if raw_idx is None else int(raw_idx)
            if idx in used:
                raise ValueError(
                    f"duplicate sample_idx={idx} for prompt_id={prompt_id!r}"
                )
            used.add(idx)
            assigned.append({**row, SAMPLE_IDX_KEY: idx})
        out[prompt_id] = sorted(assigned, key=lambda r: r[SAMPLE_IDX_KEY])
    return out

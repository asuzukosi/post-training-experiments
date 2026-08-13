"""constraint-pair builder for structured preferences.

per authored row, generate the instruction with vs without an explicit
constraint. chosen = the constrained completion. the stored prompt always
includes the constraint — otherwise dpo learns to add constraints unprompted.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

WITH_SUFFIX = "with"
WITHOUT_SUFFIX = "without"


def gen_item_id(prompt_id: str, suffix: str) -> str:
    """stable generation id: `{prompt_id}::{with|without}`."""
    prompt_id = str(prompt_id).strip()
    if not prompt_id:
        raise ValueError("prompt_id is empty")
    if suffix not in (WITH_SUFFIX, WITHOUT_SUFFIX):
        raise ValueError(f"suffix must be {WITH_SUFFIX!r} or {WITHOUT_SUFFIX!r}")
    return f"{prompt_id}::{suffix}"


def render_constrained_prompt(instruction: str, constraint: str) -> str:
    """join instruction + constraint; stored prompt must contain the constraint."""
    instruction = (instruction or "").strip()
    constraint = (constraint or "").strip()
    if not instruction:
        raise ValueError("instruction is empty")
    if not constraint:
        raise ValueError("constraint is empty")
    prompt = f"{instruction}\n\n{constraint}"
    assert_prompt_includes_constraint(prompt, constraint)
    return prompt


def assert_prompt_includes_constraint(prompt: str, constraint: str) -> None:
    """raise if the stored prompt does not contain the constraint text."""
    constraint = (constraint or "").strip()
    if not constraint:
        raise ValueError("constraint is empty")
    if constraint not in (prompt or ""):
        raise ValueError(
            "stored prompt is missing the constraint; "
            "dpo would learn to add constraints unprompted"
        )


def _row_field(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"authored row missing non-empty {key!r}: {row!r}")
    return value.strip()


def normalize_authored_row(row: Mapping[str, Any]) -> dict[str, str]:
    """require id + instruction + constraint."""
    prompt_id = row.get("id") or row.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"authored row missing non-empty id: {row!r}")
    return {
        "id": prompt_id.strip(),
        "instruction": _row_field(row, "instruction"),
        "constraint": _row_field(row, "constraint"),
    }


def load_authored_prompts(path: str | Path) -> list[dict[str, str]]:
    """load authored constraint prompts from jsonl."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"authored prompts not found: {p}")
    rows: list[dict[str, str]] = []
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid jsonl at {p}:{line_no}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"jsonl row must be an object at {p}:{line_no}")
            rows.append(normalize_authored_row(raw))
    if not rows:
        raise ValueError(f"no authored prompts in {p}")
    return rows


def build_generation_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """two gen items per authored row: constrained prompt, then instruction-only."""
    items: list[dict[str, str]] = []
    for raw in rows:
        row = normalize_authored_row(raw)
        constrained = render_constrained_prompt(row["instruction"], row["constraint"])
        items.append(
            {
                "id": gen_item_id(row["id"], WITH_SUFFIX),
                "prompt": constrained,
                "prompt_id": row["id"],
                "arm": WITH_SUFFIX,
            }
        )
        items.append(
            {
                "id": gen_item_id(row["id"], WITHOUT_SUFFIX),
                "prompt": row["instruction"],
                "prompt_id": row["id"],
                "arm": WITHOUT_SUFFIX,
            }
        )
    return items


def _require_completion(text: str, *, label: str) -> str:
    value = (text or "").strip()
    if not value:
        raise ValueError(f"{label} completion is empty")
    return value


def pair_from_completions(
    row: Mapping[str, Any],
    constrained_completion: str,
    unconstrained_completion: str,
) -> dict[str, Any]:
    """build one dpo pair; both sides share the constrained prompt."""
    authored = normalize_authored_row(row)
    prompt = render_constrained_prompt(authored["instruction"], authored["constraint"])
    chosen_text = _require_completion(constrained_completion, label="constrained")
    rejected_text = _require_completion(unconstrained_completion, label="unconstrained")
    user = {"role": "user", "content": prompt}
    return {
        "prompt": prompt,
        "prompt_id": authored["id"],
        "constraint": authored["constraint"],
        "chosen": [user, {"role": "assistant", "content": chosen_text}],
        "rejected": [dict(user), {"role": "assistant", "content": rejected_text}],
    }


def _completions_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in records:
        rec_id = rec.get("id")
        if not isinstance(rec_id, str) or not rec_id:
            raise ValueError(f"generation record missing id: {rec!r}")
        completion = rec.get("completion")
        if not isinstance(completion, str):
            raise ValueError(f"generation record missing completion: {rec_id}")
        out[rec_id] = completion
    return out


def build_structured_pairs(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """join authored rows to with/without completions."""
    by_id = _completions_by_id(records)
    pairs: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in rows:
        row = normalize_authored_row(raw)
        with_id = gen_item_id(row["id"], WITH_SUFFIX)
        without_id = gen_item_id(row["id"], WITHOUT_SUFFIX)
        if with_id not in by_id or without_id not in by_id:
            missing.append(row["id"])
            continue
        pairs.append(
            pair_from_completions(row, by_id[with_id], by_id[without_id])
        )
    if missing:
        sample = missing[:3]
        raise ValueError(
            f"missing with/without completions for {len(missing)} prompts; "
            f"examples={sample}"
        )
    return pairs

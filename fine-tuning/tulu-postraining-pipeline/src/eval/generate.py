"""resume-safe prompt completion: skip done ids, generate pending, append jsonl.

this is the eval-side counterpart of train checkpoint resume. long generation
jobs (head-to-head candidates, style probes) must survive pod kills: results land on the
volume as they finish, and a restart only pays for unfinished prompt ids.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.io import ID_KEY, PROMPT_KEY, append_jsonl, load_completed_ids
from eval.vllm_backend import vllm_generate
from prepare.paths import resolve_path

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_BATCH_SIZE = 8


def pending_items(
    items: Sequence[Mapping[str, Any]],
    completed_ids: set[str],
) -> list[Mapping[str, Any]]:
    """filter to items whose id is not in completed_ids."""
    pending: list[Mapping[str, Any]] = []
    for item in items:
        if ID_KEY not in item:
            raise ValueError(f"item missing {ID_KEY!r}: {item!r}")
        item_id = str(item[ID_KEY])
        if item_id not in completed_ids:
            pending.append(item)
    return pending


def item_id_and_prompt(item: Mapping[str, Any]) -> tuple[str, str]:
    """require non-empty id + prompt strings on an item."""
    if ID_KEY not in item:
        raise ValueError(f"item missing {ID_KEY!r}: {item!r}")
    if PROMPT_KEY not in item:
        raise ValueError(f"item missing {PROMPT_KEY!r} (id={item.get(ID_KEY)!r})")
    prompt = item[PROMPT_KEY]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"item {PROMPT_KEY!r} must be a non-empty str (id={item[ID_KEY]!r})"
        )
    return str(item[ID_KEY]), prompt


def build_record(
    item: Mapping[str, Any],
    *,
    item_id: str,
    prompt: str,
    completion: str,
    model: str,
) -> dict[str, Any]:
    """standard gen record plus any extra item fields (e.g. source)."""
    record: dict[str, Any] = {
        ID_KEY: item_id,
        PROMPT_KEY: prompt,
        "completion": completion,
        "model": model,
    }
    for k, v in item.items():
        if k not in record:
            record[k] = v
    return record


def _write_batch(
    batch: Sequence[Mapping[str, Any]],
    completions: Sequence[str],
    *,
    path: Path,
    model: str,
) -> list[dict[str, Any]]:
    if len(completions) != len(batch):
        raise RuntimeError(
            f"vllm_generate returned {len(completions)} completions "
            f"for batch size {len(batch)}"
        )
    written: list[dict[str, Any]] = []
    for item, completion in zip(batch, completions, strict=True):
        item_id, prompt = item_id_and_prompt(item)
        record = build_record(
            item,
            item_id=item_id,
            prompt=prompt,
            completion=completion,
            model=model,
        )
        append_jsonl(path, record)
        written.append(record)
    return written


def generate_incremental(
    items: Sequence[Mapping[str, Any]],
    *,
    model: str,
    output_path: str | Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """generate only unfinished prompts; append each batch to jsonl immediately.

    returns records written on this call (not the full file).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    path = resolve_path(output_path)
    completed = load_completed_ids(path)
    todo = pending_items(items, completed)
    print(
        f"vllm generate: model={model} total={len(items)} "
        f"done={len(completed)} pending={len(todo)} out={path}"
    )
    if not todo:
        return []

    written: list[dict[str, Any]] = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        prompts = [item_id_and_prompt(item)[1] for item in batch]
        completions = vllm_generate(
            prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        batch_written = _write_batch(
            batch,
            completions,
            path=path,
            model=model,
        )
        written.extend(batch_written)
        print(
            f"vllm generate: wrote {len(written)}/{len(todo)} "
            f"(batch end={min(start + batch_size, len(todo))})"
        )

    print(f"vllm generate done: wrote={len(written)} path={path}")
    return written

"""resume-safe prompt completion: skip done ids, generate pending, append jsonl.

this is the eval-side counterpart of train checkpoint resume. long generation
jobs (head-to-head candidates, style probes) must survive pod kills: results land on the
volume as they finish, and a restart only pays for unfinished prompt ids.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.eval.io import append_jsonl, load_completed_ids
from pipeline.eval.vllm_backend import vllm_generate
from pipeline.prepare.paths import resolve_path

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_BATCH_SIZE = 8

GenerateFn = Callable[[Sequence[str]], list[str]]


def pending_items(
    items: Sequence[Mapping[str, Any]],
    completed_ids: set[str],
    *,
    id_key: str = "id",
) -> list[Mapping[str, Any]]:
    """filter to items whose id is not in completed_ids."""
    pending: list[Mapping[str, Any]] = []
    for item in items:
        if id_key not in item:
            raise ValueError(f"item missing {id_key!r}: {item!r}")
        item_id = str(item[id_key])
        if item_id not in completed_ids:
            pending.append(item)
    return pending


def item_id_and_prompt(
    item: Mapping[str, Any],
    *,
    id_key: str = "id",
    prompt_key: str = "prompt",
) -> tuple[str, str]:
    """require non-empty id + prompt strings on an item."""
    if id_key not in item:
        raise ValueError(f"item missing {id_key!r}: {item!r}")
    if prompt_key not in item:
        raise ValueError(f"item missing {prompt_key!r} (id={item.get(id_key)!r})")
    prompt = item[prompt_key]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"item {prompt_key!r} must be a non-empty str (id={item[id_key]!r})"
        )
    return str(item[id_key]), prompt


def build_record(
    item: Mapping[str, Any],
    *,
    item_id: str,
    prompt: str,
    completion: str,
    model: str,
    id_key: str = "id",
    prompt_key: str = "prompt",
) -> dict[str, Any]:
    """standard gen record plus any extra item fields (e.g. source)."""
    record: dict[str, Any] = {
        "id": item_id,
        "prompt": prompt,
        "completion": completion,
        "model": model,
    }
    for k, v in item.items():
        if k not in record and k not in (id_key, prompt_key):
            record[k] = v
    return record


def _resolve_generate_fn(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    generate_fn: GenerateFn | None,
    llm: Any | None,
) -> GenerateFn:
    if generate_fn is not None:
        return generate_fn

    def _vllm(batch_prompts: Sequence[str]) -> list[str]:
        return vllm_generate(
            batch_prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            llm=llm,
        )

    return _vllm


def _write_batch(
    batch: Sequence[Mapping[str, Any]],
    completions: Sequence[str],
    *,
    path: Path,
    model: str,
    id_key: str,
    prompt_key: str,
) -> list[dict[str, Any]]:
    if len(completions) != len(batch):
        raise RuntimeError(
            f"generate_fn returned {len(completions)} completions "
            f"for batch size {len(batch)}"
        )
    written: list[dict[str, Any]] = []
    for item, completion in zip(batch, completions, strict=True):
        item_id, prompt = item_id_and_prompt(item, id_key=id_key, prompt_key=prompt_key)
        record = build_record(
            item,
            item_id=item_id,
            prompt=prompt,
            completion=completion,
            model=model,
            id_key=id_key,
            prompt_key=prompt_key,
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
    id_key: str = "id",
    prompt_key: str = "prompt",
    generate_fn: GenerateFn | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """generate only unfinished prompts; append each batch to jsonl immediately.

    returns records written on this call (not the full file). pass `generate_fn`
    to avoid loading vllm (tests / mocks).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    path = resolve_path(output_path)
    completed = load_completed_ids(path, id_key=id_key)
    todo = pending_items(items, completed, id_key=id_key)
    print(
        f"vllm generate: model={model} total={len(items)} "
        f"done={len(completed)} pending={len(todo)} out={path}"
    )
    if not todo:
        return []

    run_generate = _resolve_generate_fn(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        generate_fn=generate_fn,
        llm=llm,
    )

    written: list[dict[str, Any]] = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        prompts = [
            item_id_and_prompt(item, id_key=id_key, prompt_key=prompt_key)[1]
            for item in batch
        ]
        batch_written = _write_batch(
            batch,
            run_generate(prompts),
            path=path,
            model=model,
            id_key=id_key,
            prompt_key=prompt_key,
        )
        written.extend(batch_written)
        print(
            f"vllm generate: wrote {len(written)}/{len(todo)} "
            f"(batch end={min(start + batch_size, len(todo))})"
        )

    print(f"vllm generate done: wrote={len(written)} path={path}")
    return written

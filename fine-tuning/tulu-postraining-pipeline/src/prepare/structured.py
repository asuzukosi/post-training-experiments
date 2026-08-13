"""prepare structured preference pairs from authored constraint prompts."""
from __future__ import annotations
from pathlib import Path
from typing import Any

from data_tools.structured import (
    build_generation_items,
    build_structured_pairs,
    load_authored_prompts,
)
from prepare.io import save_rows
from prepare.paths import resolve_path


def resolve_generator_model(
    cfg: dict[str, Any],
    generator_model: str | None,
) -> str:
    """require a generator checkpoint or hub id."""
    model = generator_model if generator_model is not None else cfg.get("generator_model")
    if model:
        return str(model)
    raise ValueError(
        "generator_model is required; pass --generator-model or set "
        "generator_model in configs/structured.yaml"
    )


def prepare_structured(
    cfg: dict[str, Any],
    *,
    prompts_path: str | Path | None = None,
    generator_model: str | None = None,
) -> Path:
    """load authored prompts, generate with/without constraint, save dpo pairs."""
    from eval.generate import generate_incremental
    from eval.io import load_jsonl

    authored_path = resolve_path(prompts_path or cfg["prompts_path"])
    rows = load_authored_prompts(authored_path)
    items = build_generation_items(rows)
    model = resolve_generator_model(cfg, generator_model)
    gen_path = resolve_path(cfg["generations_path"])
    print(
        f"structured gen: prompts={len(rows)} items={len(items)} "
        f"model={model} out={gen_path}"
    )
    generate_incremental(
        items,
        model=model,
        output_path=gen_path,
        max_tokens=int(cfg.get("max_tokens", 1024)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        batch_size=int(cfg.get("batch_size", 8)),
    )
    records = load_jsonl(gen_path)
    pairs = build_structured_pairs(rows, records)
    print(f"structured pairs: {len(pairs)}")
    return save_rows(pairs, cfg["processed_path"])

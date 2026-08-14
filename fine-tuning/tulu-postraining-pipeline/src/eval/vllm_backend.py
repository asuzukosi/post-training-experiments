"""thin vllm wrapper used by generate_incremental.

the engine is cached across calls, and only one is ever live.

both halves matter. `generate_incremental` calls `vllm_generate` once per *write*
batch — batching there buys resume granularity, not throughput — so constructing an
`LLM` per call pays the full weight-load + kv-profile + cuda-graph cost every 8
prompts. and `run_head_to_head` alternates two models, so a cache that kept both
resident would OOM: an `LLM` reserves gpu_memory_utilization (0.9 by default) of the
card up front, and two of those do not fit on one gpu.

vllm does its own continuous batching internally, so handing it a whole batch at once
is all the batching that is wanted here.
"""
from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

# (model, engine) for the single live engine, or None
_ENGINE: tuple[str, Any] | None = None


def release_engine() -> None:
    """drop the live engine and give its gpu memory back."""
    global _ENGINE
    if _ENGINE is None:
        return
    print(f"vllm: releasing engine model={_ENGINE[0]}")
    _ENGINE = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_engine(model: str) -> Any:
    """the live LLM for `model`, loading it — and evicting any other — on first use."""
    global _ENGINE
    if _ENGINE is not None and _ENGINE[0] == model:
        return _ENGINE[1]
    release_engine()
    from vllm import LLM

    print(f"vllm: loading engine model={model}")
    _ENGINE = (model, LLM(model=model))
    return _ENGINE[1]


def vllm_generate(
    prompts: Sequence[str],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """run vllm generate for one batch."""
    from vllm import SamplingParams

    engine = get_engine(model)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = engine.generate(list(prompts), sampling)
    completions: list[str] = []
    for out in outputs:
        if not out.outputs:
            completions.append("")
        else:
            completions.append(out.outputs[0].text)
    return completions

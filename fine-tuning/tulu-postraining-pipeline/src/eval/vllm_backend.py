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
from collections.abc import Callable, Sequence
from typing import Any

# (key, engine) for the single live engine, or None
_LIVE: tuple[str, Any] | None = None


def release_engine() -> None:
    """drop the live engine and give its gpu memory back."""
    global _LIVE
    if _LIVE is None:
        return
    print(f"vllm: releasing engine {_LIVE[0]}")
    _LIVE = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def live_engine(key: str, factory: Callable[[], Any]) -> Any:
    """the one live engine for `key`, building it — and evicting any other — on demand.

    every engine here reserves gpu_memory_utilization (0.9 by default) of the card up
    front, so exactly one may exist at a time. that constraint spans code paths, not just
    models: `run_head_to_head` generates with model a, then model b, then loads a judge,
    all in one process. keying generation and judging through the same registry is what
    makes "sequential on one gpu" actually sequential, rather than three engines racing
    for the same card.
    """
    global _LIVE
    if _LIVE is not None and _LIVE[0] == key:
        return _LIVE[1]
    release_engine()
    print(f"vllm: loading engine {key}")
    _LIVE = (key, factory())
    return _LIVE[1]


def get_engine(model: str) -> Any:
    """the live LLM for `model`, loading it — and evicting any other — on first use."""

    def _build() -> Any:
        from vllm import LLM

        return LLM(model=model)

    return live_engine(f"generate:{model}", _build)


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

"""thin vllm wrapper used by generate_incremental."""
from __future__ import annotations

from collections.abc import Sequence


def vllm_generate(
    prompts: Sequence[str],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """run vllm generate for one batch."""
    from vllm import LLM, SamplingParams

    engine = LLM(model=model)
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

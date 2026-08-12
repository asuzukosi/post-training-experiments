"""local pairwise judge: position-swapped, temp 0, in-process vllm.

items need: id, prompt, completion_a, completion_b (optional model_a/model_b/run).
for each pending id we ask the judge twice (a-then-b and b-then-a), map both
verdicts onto model labels A/B/tie, then aggregate. no openai server.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pipeline.eval.generate import pending_items
from pipeline.eval.io import append_jsonl, load_completed_ids
from pipeline.eval.vllm_backend import vllm_generate
from pipeline.prepare.paths import resolve_path

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE_MAX_TOKENS = 16
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_TOP_P = 1.0
DEFAULT_JUDGE_BATCH_SIZE = 4  # pairs per batch; each pair => 2 judge calls

Verdict = Literal["1", "2", "tie"]
Winner = Literal["A", "B", "tie"]
GenerateFn = Callable[[Sequence[str]], list[str]]

_VERDICT_RE = re.compile(
    r"\b(tie|1|2)\b",
    re.IGNORECASE,
)


def judgment_id(prompt_id: str, *, run: int = 1) -> str:
    """stable id for resume across head-to-head repeats."""
    return f"{prompt_id}__r{run}"


def build_pairwise_user_message(
    prompt: str,
    response_1: str,
    response_2: str,
) -> str:
    """user message asking for a single verdict token: 1, 2, or tie."""
    return (
        "you are comparing two assistant responses to the same user prompt.\n"
        "choose which response is better overall (helpfulness, correctness, clarity).\n"
        "if they are equal quality, answer tie.\n\n"
        f"user prompt:\n{prompt}\n\n"
        f"assistant 1:\n{response_1}\n\n"
        f"assistant 2:\n{response_2}\n\n"
        "reply with exactly one of: 1, 2, tie"
    )


def render_judge_prompt(user_message: str) -> str:
    """qwen-style chatml string with generation prompt (no tokenizer required)."""
    system = (
        "you are a careful, unbiased judge of assistant responses. "
        "follow the output format exactly."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_pairwise_verdict(text: str) -> Verdict:
    """parse judge text into 1 / 2 / tie; prefer first clear match."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "tie"
    # exact short answers first
    low = cleaned.lower()
    if low in ("1", "2", "tie"):
        return low  # type: ignore[return-value]
    match = _VERDICT_RE.search(cleaned)
    if match is None:
        return "tie"
    token = match.group(1).lower()
    if token in ("1", "2", "tie"):
        return token  # type: ignore[return-value]
    return "tie"


def verdict_to_winner(verdict: Verdict, *, first_is_a: bool) -> Winner:
    """map displayed assistant 1/2 onto model labels A/B."""
    if verdict == "tie":
        return "tie"
    if verdict == "1":
        return "A" if first_is_a else "B"
    # verdict == "2"
    return "B" if first_is_a else "A"


def aggregate_winner(order_ab: Winner, order_ba: Winner) -> Winner:
    """
    both orders must agree; otherwise tie (position-bias / disagreement).
    this is to prevent the case where the judge is biased and always chooses the same model.
    """
    if order_ab == order_ba and order_ab in ("A", "B", "tie"):
        return order_ab
    if order_ab == order_ba:
        return "tie"
    return "tie"


def _require_pair_fields(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if "id" not in item:
        raise ValueError(f"judge item missing 'id': {item!r}")
    for key in ("prompt", "completion_a", "completion_b"):
        if key not in item:
            raise ValueError(f"judge item missing {key!r} (id={item.get('id')!r})")
        val = item[key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"judge item {key!r} must be a non-empty str (id={item.get('id')!r})"
            )
    return (
        str(item["id"]),
        item["prompt"],
        item["completion_a"],
        item["completion_b"],
    )


def build_judge_prompts_for_item(item: Mapping[str, Any]) -> tuple[str, str]:
    """return (order_ab_prompt, order_ba_prompt) rendered for the judge lm."""
    _, prompt, completion_a, completion_b = _require_pair_fields(item)
    ab = render_judge_prompt(
        build_pairwise_user_message(prompt, completion_a, completion_b)
    )
    ba = render_judge_prompt(
        build_pairwise_user_message(prompt, completion_b, completion_a)
    )
    return ab, ba


def build_judge_record(
    item: Mapping[str, Any],
    *,
    order_ab: Winner,
    order_ba: Winner,
    winner: Winner,
    judge_model: str,
    raw_ab: str,
    raw_ba: str,
) -> dict[str, Any]:
    item_id, prompt, completion_a, completion_b = _require_pair_fields(item)
    record: dict[str, Any] = {
        "id": item_id,
        "prompt": prompt,
        "completion_a": completion_a,
        "completion_b": completion_b,
        "model_a": item.get("model_a"),
        "model_b": item.get("model_b"),
        "judge_model": judge_model,
        "order_ab": order_ab,
        "order_ba": order_ba,
        "winner": winner,
        "raw_ab": raw_ab,
        "raw_ba": raw_ba,
    }
    if "run" in item:
        record["run"] = item["run"]
    for k, v in item.items():
        if k not in record:
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


def judge_incremental(
    items: Sequence[Mapping[str, Any]],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    output_path: str | Path,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    top_p: float = DEFAULT_JUDGE_TOP_P,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    generate_fn: GenerateFn | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """score pending pairs with position swap; append jsonl per finished pair.

    each item uses two judge calls (ab and ba). returns records written this call.
    pass `generate_fn` to avoid loading vllm (tests / mocks).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    path = resolve_path(output_path)
    completed = load_completed_ids(path)
    todo = pending_items(items, completed)
    print(
        f"vllm judge: model={judge_model} total={len(items)} "
        f"done={len(completed)} pending={len(todo)} out={path}"
    )
    if not todo:
        return []

    run_generate = _resolve_generate_fn(
        model=judge_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        generate_fn=generate_fn,
        llm=llm,
    )

    written: list[dict[str, Any]] = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        prompts: list[str] = []
        for item in batch:
            ab, ba = build_judge_prompts_for_item(item)
            prompts.append(ab)
            prompts.append(ba)

        outs = run_generate(prompts)
        if len(outs) != len(prompts):
            raise RuntimeError(
                f"generate_fn returned {len(outs)} texts for {len(prompts)} prompts"
            )

        for i, item in enumerate(batch):
            raw_ab = outs[2 * i]
            raw_ba = outs[2 * i + 1]
            v_ab = parse_pairwise_verdict(raw_ab)
            v_ba = parse_pairwise_verdict(raw_ba)
            order_ab = verdict_to_winner(v_ab, first_is_a=True)
            order_ba = verdict_to_winner(v_ba, first_is_a=False)
            winner = aggregate_winner(order_ab, order_ba)
            record = build_judge_record(
                item,
                order_ab=order_ab,
                order_ba=order_ba,
                winner=winner,
                judge_model=judge_model,
                raw_ab=raw_ab,
                raw_ba=raw_ba,
            )
            append_jsonl(path, record)
            written.append(record)

        print(
            f"vllm judge: wrote {len(written)}/{len(todo)} "
            f"(batch end={min(start + batch_size, len(todo))})"
        )

    print(f"vllm judge done: wrote={len(written)} path={path}")
    return written

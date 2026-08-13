"""judgearena pairwise wrapper: local vllm qwen, position-swapped, temp 0.

items need: id, prompt, completion_a, completion_b (optional model_a/model_b/run).
does not reimplement judge prompts or verdict parsing — judgearena owns those.
no openai server.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from eval.generate import pending_items
from eval.io import append_jsonl, load_completed_ids
from prepare.paths import resolve_path

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_BATCH_SIZE = 4

FIRST_MODEL = "A"
SECOND_MODEL = "B"
MODEL_TIE = "tie"

Winner = Literal[FIRST_MODEL, SECOND_MODEL, MODEL_TIE]


def judgment_id(prompt_id: str, *, run: int = 1) -> str:
    """stable id for resume across head-to-head repeats."""
    return f"{prompt_id}__r{run}"


def judgearena_model_id(model: str) -> str:
    """judgearena backend id: `VLLM/<hf-or-path>`."""
    raw = (model or "").strip()
    if not raw:
        raise ValueError("judge model is empty")
    if raw.startswith("VLLM/"):
        return raw
    return f"VLLM/{raw}"


def preference_to_winner(pref: float | None) -> Winner:
    """map judgearena P(B wins) in {0, 0.5, 1} onto A/B/tie."""
    if pref is None:
        return MODEL_TIE
    value = float(pref) # if there is a parsing error, we treat it as a tie
    if value != value:
        return MODEL_TIE
    if value < 0.5:
        return FIRST_MODEL
    if value > 0.5:
        return SECOND_MODEL
    return MODEL_TIE


def aggregate_winner(order_ab: Winner, order_ba: Winner) -> Winner:
    """both orders must agree; otherwise tie (position disagreement)."""
    if order_ab == order_ba and order_ab in (FIRST_MODEL, SECOND_MODEL, MODEL_TIE):
        return order_ab
    return MODEL_TIE


def _require_pair_fields(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """fail fast unless the item has id, prompt, and both completions as non-empty strings."""
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


def preference_field(row: Mapping[str, Any], key: str) -> float | None:
    """read a judgearena pref key as float; missing or null becomes none (nan is kept)."""
    if key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    return float(value)


def build_judge_record(
    item: Mapping[str, Any],
    *,
    score: Mapping[str, Any],
    judge_model: str,
) -> dict[str, Any]:
    item_id, prompt, completion_a, completion_b = _require_pair_fields(item)
    pref_ab = preference_field(score, "pref_ab")
    pref_ba = preference_field(score, "pref_ba")
    order_ab = preference_to_winner(pref_ab)
    order_ba = preference_to_winner(pref_ba)
    record: dict[str, Any] = {
        "id": item_id,
        "prompt": prompt,
        "completion_a": completion_a,
        "completion_b": completion_b,
        "model_a": item.get("model_a"),
        "model_b": item.get("model_b"),
        "judge_model": judge_model,
        "judge_backend": "judgearena",
        "judge_backend_id": judgearena_model_id(judge_model),
        "pref_ab": pref_ab,
        "pref_ba": pref_ba,
        "order_ab": order_ab,
        "order_ba": order_ba,
        "winner": aggregate_winner(order_ab, order_ba),
        "raw_ab": score.get("raw_ab"), # raw ab value before softmax   
        "raw_ba": score.get("raw_ba"), # raw ba value before softmax
    }
    if "run" in item:
        record["run"] = item["run"]
    for k, v in item.items(): # add other metadata from the item
        if k not in record:
            record[k] = v
    return record


def score_with_judgearena(
    batch: Sequence[Mapping[str, Any]],
    *,
    judge_model: str,
    temperature: float,
) -> list[Mapping[str, Any]]:
    """
    annotate both orders via judgearena.annotate_battles, then parse with PairScore.
    judge_and_parse_prefs is git-main only; 0.1.0 (the pypi pin) exposes annotate_battles.
    swap_mode=both is two annotate_battles calls with a/b swapped, same as the 0.1.0 cli.
    """
    from judgearena.evaluate import PairScore, annotate_battles
    from judgearena.utils import make_model

    llm = make_model(judgearena_model_id(judge_model), temperature=float(temperature))
    instructions = [str(item["prompt"]) for item in batch]
    completions_a = [str(item["completion_a"]) for item in batch]
    completions_b = [str(item["completion_b"]) for item in batch]

    # get results for both orders to prevent position bias
    annotations = annotate_battles( # get result for model A vs model B
        judge_chat_model=llm,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
    )
    reversed_anns = annotate_battles( # get result for model B vs model A
        judge_chat_model=llm,
        instructions=instructions,
        completions_A=completions_b,
        completions_B=completions_a,
    )
    if len(reversed_anns) != len(annotations):
        raise RuntimeError(
            f"judgearena reversed pass returned {len(reversed_anns)} "
            f"for {len(annotations)} annotations"
        )
    parser = PairScore() # pair score parser to parse the judgearena results and give the preference score
    out: list[Mapping[str, Any]] = []
    for ann, rev in zip(annotations, reversed_anns, strict=True):
        pref_ab = parser.parse_model_raw(ann.judge_completion) # get the float value of the preference for model A vs model B i.e p(B wins)
        pref_ba_displayed = parser.parse_model_raw(rev.judge_completion) # get the float value of the preference for model B vs model A p(A wins)
        pref_ba = None if pref_ba_displayed is None else 1.0 - float(pref_ba_displayed) # have to flip becuse b is currently in secon possition 
        out.append(
            {
                "pref_ab": pref_ab,
                "pref_ba": pref_ba,
                "raw_ab": ann.judge_completion,
                "raw_ba": rev.judge_completion,
            }
        )
    return out


def judge_incremental(
    items: Sequence[Mapping[str, Any]],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    output_path: str | Path,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """score pending pairs via judgearena; append jsonl per finished pair."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for item in items:
        _require_pair_fields(item) # fail fast unless the item has id, prompt, and both completions as non-empty strings

    path = resolve_path(output_path)
    completed = load_completed_ids(path) # load the completed ids from the output path
    todo = pending_items(items, completed) # get the pending items
    backend_id = judgearena_model_id(judge_model) # get the judgearena model id
    print(
        f"judgearena: model={backend_id} temp={temperature:g} total={len(items)} "
        f"done={len(completed)} pending={len(todo)} out={path}"
    )
    if not todo:
        return []

    written: list[dict[str, Any]] = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        scores = score_with_judgearena(
            batch,
            judge_model=judge_model,
            temperature=temperature,
        )
        if len(scores) != len(batch):
            raise RuntimeError(
                f"judgearena returned {len(scores)} scores for batch size {len(batch)}"
            )
        for item, score in zip(batch, scores, strict=True):
            record = build_judge_record(item, score=score, judge_model=judge_model)
            append_jsonl(path, record)
            written.append(record)
        print(
            f"judgearena: wrote {len(written)}/{len(todo)} "
            f"(batch end={min(start + batch_size, len(todo))})"
        )

    print(f"judgearena done: wrote={len(written)} path={path}")
    return written

"""single-elim tournament: pair, judge, advance winners."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.bon.candidates import COMPLETION_KEY, PROMPT_ID_KEY, SAMPLE_IDX_KEY
from eval.io import ID_KEY, PROMPT_KEY, load_jsonl
from eval.judge import (
    DEFAULT_JUDGE_BATCH_SIZE,
    DEFAULT_JUDGE_TEMPERATURE,
    FIRST_MODEL,
    MODEL_TIE,
    SECOND_MODEL,
    judge_incremental,
)
from prepare.paths import resolve_path

DEFAULT_BON_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def pair_id(prompt_id: str, *, round_idx: int, sample_a: int, sample_b: int) -> str:
    """stable id so a restarted tournament skips finished comparisons."""
    return f"{prompt_id}__t{round_idx}__{sample_a}v{sample_b}"


def sort_pairs_by_length(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """shortest-max-length first so a judge batch pads less."""
    return sorted(
        (dict(p) for p in pairs),
        key=lambda p: max(len(p["completion_a"]), len(p["completion_b"])),
    )


def pick_winner_candidate(
    record: Mapping[str, Any],
    cand_a: Mapping[str, Any],
    cand_b: Mapping[str, Any],
) -> dict[str, Any]:
    """map a judgearena pair record onto the winning candidate."""
    winner = record["winner"]
    if winner == SECOND_MODEL:
        return dict(cand_b)
    if winner == FIRST_MODEL:
        return dict(cand_a)
    if winner != MODEL_TIE:
        raise ValueError(f"unexpected winner {winner!r}")
    len_a = len(cand_a[COMPLETION_KEY])
    len_b = len(cand_b[COMPLETION_KEY])
    if len_b < len_a:
        return dict(cand_b)
    if len_a < len_b:
        return dict(cand_a)
    if cand_b[SAMPLE_IDX_KEY] < cand_a[SAMPLE_IDX_KEY]:
        return dict(cand_b)
    return dict(cand_a)


def build_round_pairs(
    remaining: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    round_idx: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """pair adjacent sample_idx; odd leftover gets a bye (last after sort)."""
    pairs: list[dict[str, Any]] = []
    byes: dict[str, dict[str, Any]] = {}
    for prompt_id, cands in remaining.items():
        ordered = sorted(cands, key=lambda c: int(c[SAMPLE_IDX_KEY]))
        if len(ordered) <= 1:
            continue
        if len(ordered) % 2 == 1:
            byes[prompt_id] = dict(ordered[-1])
            ordered = ordered[:-1]
        prompt = ordered[0][PROMPT_KEY]
        for i in range(0, len(ordered), 2):
            a = ordered[i]
            b = ordered[i + 1]
            idx_a = int(a[SAMPLE_IDX_KEY])
            idx_b = int(b[SAMPLE_IDX_KEY])
            pairs.append(
                {
                    ID_KEY: pair_id(
                        prompt_id,
                        round_idx=round_idx,
                        sample_a=idx_a,
                        sample_b=idx_b,
                    ),
                    PROMPT_KEY: prompt,
                    "completion_a": a[COMPLETION_KEY],
                    "completion_b": b[COMPLETION_KEY],
                    PROMPT_ID_KEY: prompt_id,
                    "round": round_idx,
                    "sample_idx_a": idx_a,
                    "sample_idx_b": idx_b,
                }
            )
    return sort_pairs_by_length(pairs), byes


def _index_by_sample(
    cands: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {int(c[SAMPLE_IDX_KEY]): dict(c) for c in cands}


def apply_round_results(
    remaining: Mapping[str, Sequence[Mapping[str, Any]]],
    pairs: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    byes: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """advance winners (+ byes) to the next remaining set."""
    next_remaining: dict[str, list[dict[str, Any]]] = {
        prompt_id: [dict(c) for c in cands]
        for prompt_id, cands in remaining.items()
        if len(cands) == 1
    }
    winners_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        prompt_id = str(pair[PROMPT_ID_KEY])
        rec = records_by_id.get(str(pair[ID_KEY])) # get the judgearena results for the pair
        if rec is None:
            raise ValueError(f"missing judgment for pair {pair[ID_KEY]!r}")
        by_sample = _index_by_sample(remaining[prompt_id]) # get the candidates for the prompt_id
        cand_a = by_sample[int(pair["sample_idx_a"])]
        cand_b = by_sample[int(pair["sample_idx_b"])]
        winners_by_prompt.setdefault(prompt_id, []).append(
            pick_winner_candidate(rec, cand_a, cand_b) # add the winners for this round based on the judgearena results to the prompt_id
        )
    for prompt_id, bye in byes.items(): # add the bye candidates to the winners by prompt
        winners_by_prompt.setdefault(prompt_id, []).append(dict(bye))
    for prompt_id, winners in winners_by_prompt.items(): # sort the winners by sample index
        next_remaining[prompt_id] = sorted(
            winners, key=lambda c: int(c[SAMPLE_IDX_KEY])
        )
    return next_remaining


def select_top1(
    candidates_by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    judge_model: str = DEFAULT_BON_JUDGE_MODEL,
    judge_path: str | Path,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> dict[str, dict[str, Any]]:
    """run the tournament; return prompt_id -> winning candidate."""
    if not candidates_by_prompt:
        raise ValueError("no candidates to select from")
    remaining: dict[str, list[dict[str, Any]]] = {
        prompt_id: [dict(c) for c in cands]
        for prompt_id, cands in candidates_by_prompt.items()
    }
    for prompt_id, cands in remaining.items():
        if not cands:
            raise ValueError(f"prompt_id {prompt_id!r} has no candidates")

    path = resolve_path(judge_path)
    round_idx = 1
    while any(len(cands) > 1 for cands in remaining.values()):
        pairs, byes = build_round_pairs(remaining, round_idx=round_idx)
        print(
            f"bon tournament: round={round_idx} pairs={len(pairs)} "
            f"byes={len(byes)} judge={judge_model}"
        )
        judge_incremental(
            pairs,
            judge_model=judge_model,
            output_path=path,
            temperature=temperature,
            batch_size=batch_size,
        )
        records_by_id = {str(r[ID_KEY]): r for r in load_jsonl(path)}
        remaining = apply_round_results(remaining, pairs, records_by_id, byes)
        round_idx += 1

    winners = {
        prompt_id: cands[0] for prompt_id, cands in remaining.items()
    }
    print(f"bon tournament done: prompts={len(winners)} path={path}")
    return winners

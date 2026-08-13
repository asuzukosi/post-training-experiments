"""write top-1 selections as an rs-sft conversational dataset."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.bon.candidates import (
    COMPLETION_KEY,
    PROMPT_ID_KEY,
    SAMPLE_IDX_KEY,
    group_candidates,
)
from eval.bon.tournament import DEFAULT_BON_JUDGE_MODEL, select_top1
from eval.io import PROMPT_KEY, append_jsonl, load_jsonl
from eval.judge import DEFAULT_JUDGE_BATCH_SIZE, DEFAULT_JUDGE_TEMPERATURE
from prepare.io import save_rows
from prepare.paths import resolve_path


def build_rs_sft_row(candidate: Mapping[str, Any], *, n_candidates: int) -> dict[str, Any]:
    """conversational sft row: user prompt + selected assistant completion."""
    return {
        PROMPT_ID_KEY: candidate[PROMPT_ID_KEY],
        SAMPLE_IDX_KEY: candidate[SAMPLE_IDX_KEY],
        "n_candidates": n_candidates,
        "source": "rs_sft",
        "messages": [
            {"role": "user", "content": candidate[PROMPT_KEY]},
            {"role": "assistant", "content": candidate[COMPLETION_KEY]},
        ],
    }


def run_bon_selection(
    *,
    generations_path: str | Path,
    output_dir: str | Path,
    processed_path: str | Path,
    judge_model: str = DEFAULT_BON_JUDGE_MODEL,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> Path:
    """load n-sample gens, tournament to top-1, write rs-sft dataset."""
    gens = load_jsonl(generations_path)
    if not gens:
        raise ValueError(f"no generations in {generations_path}")
    grouped = group_candidates(gens)
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_path = out_dir / "bon_pairs.jsonl"
    winners = select_top1(
        grouped,
        judge_model=judge_model,
        judge_path=judge_path,
        temperature=temperature,
        batch_size=batch_size,
    )
    selections_path = out_dir / "bon_selections.jsonl"
    if selections_path.exists():
        selections_path.unlink()
    rows: list[dict[str, Any]] = []
    for prompt_id in sorted(winners):
        winner = winners[prompt_id]
        n = len(grouped[prompt_id])
        row = build_rs_sft_row(winner, n_candidates=n)
        append_jsonl(selections_path, {**winner, "n_candidates": n})
        rows.append(row)
    print(f"bon selections: n={len(rows)} out={selections_path}")
    return save_rows(rows, processed_path)

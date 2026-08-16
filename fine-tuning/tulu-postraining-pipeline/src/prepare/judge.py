"""the frozen judging prompt set — test_prefs, minus everything ppo trains on.

every judged comparison in the programme runs on this one file, so base-vs-sft and
sft-vs-dpo are measured on identical ground. it is written once and not resampled.

size is decided by what is left rather than by a target: ppo takes 1,500 of the 1,981
clean test_prefs prompts, and judging ppo on prompts it was rl-trained on would inflate
its win-rate, so the judging set is the remainder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.io import ID_KEY, PROMPT_KEY
from prepare.decontam import build_eval_decontam_bank
from prepare.io import load_processed_rows, load_ultrafeedback
from prepare.paths import resolve_path

DEFAULT_JUDGE_PROMPTS_PATH = "data/processed/eval_prompts.jsonl"


def _ppo_exclude_ids(
    ppo_cfg: dict[str, Any],
    rows,
    ngram_bank: set[str] | None = None,
) -> set[str]:
    """prompt_ids from the processed ppo pool, or rebuilt in-memory with the same seed.

    the rebuild has to match ppo prep exactly — same seed, same bank — or the ids held
    back here are not the ids ppo will train on.
    """
    from data_tools import build_ultrafeedback_prompt_pool, prompt_ids_of

    ppo_path = resolve_path(ppo_cfg["processed_path"])
    if ppo_path.exists():
        print(f"loading ppo exclude ids from {ppo_path}")
        return prompt_ids_of(load_processed_rows(ppo_path))

    print("ppo processed path missing; rebuilding the ppo pool in-memory for exclude ids")
    return prompt_ids_of(
        build_ultrafeedback_prompt_pool(
            rows,
            num_prompts=int(ppo_cfg["num_prompts"]),
            seed=int(ppo_cfg["seed"]),
            ngram_bank=ngram_bank,
        )
    )


def prepare_judge_prompts(
    ppo_cfg: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    num_prompts: int | None = None,
    skip_decontam: bool = False,
) -> Path:
    """write `{id, prompt}` jsonl for `scripts/eval/head_to_head.py --prompts`."""
    from data_tools import build_ultrafeedback_prompt_pool

    rows = load_ultrafeedback(ppo_cfg, default_split="test_prefs")
    ngram_bank = None if skip_decontam else build_eval_decontam_bank()
    exclude = _ppo_exclude_ids(ppo_cfg, rows, ngram_bank)

    seed = int(ppo_cfg["seed"])
    print(f"building judge prompts seed={seed} exclude={len(exclude)} take={num_prompts or 'all'}")
    pool = build_ultrafeedback_prompt_pool(
        rows,
        num_prompts=num_prompts,
        seed=seed,
        ngram_bank=ngram_bank,
        exclude_prompt_ids=exclude,
    )

    overlap = {p["prompt_id"] for p in pool} & exclude
    if overlap:
        raise ValueError(f"judge prompts overlap the ppo pool ({len(overlap)} ids)")
    print("asserted judge prompts disjoint from the ppo pool")

    out = resolve_path(output_path or DEFAULT_JUDGE_PROMPTS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps({ID_KEY: row["prompt_id"], PROMPT_KEY: row["prompt"]}) + "\n")
    print(f"wrote {len(pool)} judge prompts -> {out}")
    return out

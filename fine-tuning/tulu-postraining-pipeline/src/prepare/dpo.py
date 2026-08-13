"""prepare the ultrafeedback dpo pair subset (disjoint from rm)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.io import load_processed_rows, load_ultrafeedback, save_rows
from prepare.paths import resolve_path


def _rm_exclude_ids(rm_cfg: dict[str, Any], train_rows) -> set[str]:
    """load prompt_ids from processed rm, or rebuild in-memory with the same seed."""
    from data_tools import build_ultrafeedback_rm_subset, prompt_ids_of

    rm_path = resolve_path(rm_cfg["processed_path"])
    if rm_path.exists():
        print(f"loading rm exclude ids from {rm_path}")
        return prompt_ids_of(load_processed_rows(rm_path))

    print("rm processed path missing; sampling rm subset in-memory for exclude ids")
    rm_rows = build_ultrafeedback_rm_subset(
        train_rows,
        num_pairs=int(rm_cfg["num_pairs"]),
        seed=int(rm_cfg["seed"]),
    )
    return prompt_ids_of(rm_rows)


def prepare_dpo(dpo_cfg: dict[str, Any], rm_cfg: dict[str, Any]) -> Path:
    from data_tools import (
        assert_disjoint_prompt_ids,
        build_ultrafeedback_dpo_subset,
    )

    rows = load_ultrafeedback(dpo_cfg, default_split="train_prefs")
    exclude = _rm_exclude_ids(rm_cfg, rows)
    num_pairs = int(dpo_cfg["num_pairs"])
    seed = int(dpo_cfg["seed"])
    print(f"building dpo subset num_pairs={num_pairs} seed={seed} exclude={len(exclude)}")
    subset = build_ultrafeedback_dpo_subset(
        rows,
        exclude_prompt_ids=exclude,
        num_pairs=num_pairs,
        seed=seed,
    )
    rm_path = resolve_path(rm_cfg["processed_path"])
    if rm_path.exists():
        assert_disjoint_prompt_ids(load_processed_rows(rm_path), subset)
        print("asserted dpo disjoint from on-disk rm")
    return save_rows(subset, dpo_cfg["processed_path"])

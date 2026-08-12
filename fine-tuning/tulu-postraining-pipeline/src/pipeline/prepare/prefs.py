"""prepare ultrafeedback rm / dpo / ppo subsets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.prepare.io import load_processed_rows, save_rows
from pipeline.prepare.paths import resolve_path


def _load_ultrafeedback(cfg: dict[str, Any], *, default_split: str):
    from datasets import load_dataset

    dataset_id = cfg["dataset"]
    split = cfg.get("split", default_split)
    print(f"loading ultrafeedback: {dataset_id} split={split}")
    ds = load_dataset(dataset_id, split=split)
    print(f"ultrafeedback {split} rows: {len(ds)}")
    return ds


def prepare_rm(cfg: dict[str, Any]) -> Path:
    from pipeline.data_tools import build_ultrafeedback_rm_subset

    rows = _load_ultrafeedback(cfg, default_split="train_prefs")
    num_pairs = int(cfg["num_pairs"])
    seed = int(cfg["seed"])
    print(f"building rm subset num_pairs={num_pairs} seed={seed}")
    subset = build_ultrafeedback_rm_subset(rows, num_pairs=num_pairs, seed=seed)
    return save_rows(subset, cfg["processed_path"])


def _rm_exclude_ids(rm_cfg: dict[str, Any], train_rows) -> set[str]:
    """load prompt_ids from processed rm, or rebuild in-memory with the same seed."""
    from pipeline.data_tools import build_ultrafeedback_rm_subset, prompt_ids_of

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
    from pipeline.data_tools import (
        assert_disjoint_prompt_ids,
        build_ultrafeedback_dpo_subset,
    )

    rows = _load_ultrafeedback(dpo_cfg, default_split="train_prefs")
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


def prepare_ppo(cfg: dict[str, Any]) -> Path:
    from pipeline.data_tools import build_ultrafeedback_ppo_prompts

    ds = _load_ultrafeedback(cfg, default_split="test_prefs")
    num_prompts = int(cfg["num_prompts"])
    seed = int(cfg["seed"])
    print(f"building ppo prompts num_prompts={num_prompts} seed={seed}")
    subset = build_ultrafeedback_ppo_prompts(ds, num_prompts=num_prompts, seed=seed)
    return save_rows(subset, cfg["processed_path"])

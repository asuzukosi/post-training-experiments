"""prepare the ultrafeedback rm pair subset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.io import load_ultrafeedback, save_rows


def prepare_rm(cfg: dict[str, Any]) -> Path:
    from data_tools import build_ultrafeedback_rm_subset

    rows = load_ultrafeedback(cfg, default_split="train_prefs")
    num_pairs = int(cfg["num_pairs"])
    seed = int(cfg["seed"])
    print(f"building rm subset num_pairs={num_pairs} seed={seed}")
    subset = build_ultrafeedback_rm_subset(rows, num_pairs=num_pairs, seed=seed)
    return save_rows(subset, cfg["processed_path"])

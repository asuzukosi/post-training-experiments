"""prepare the ultrafeedback ppo prompt pool."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.io import load_ultrafeedback, save_rows


def prepare_ppo(cfg: dict[str, Any]) -> Path:
    from data_tools import build_ultrafeedback_ppo_prompts

    ds = load_ultrafeedback(cfg, default_split="test_prefs")
    num_prompts = int(cfg["num_prompts"])
    seed = int(cfg["seed"])
    print(f"building ppo prompts num_prompts={num_prompts} seed={seed}")
    subset = build_ultrafeedback_ppo_prompts(ds, num_prompts=num_prompts, seed=seed)
    return save_rows(subset, cfg["processed_path"])

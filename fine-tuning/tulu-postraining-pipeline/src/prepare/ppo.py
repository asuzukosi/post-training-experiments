"""prepare the ultrafeedback ppo prompt pool."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.decontam import build_eval_decontam_bank
from prepare.io import load_ultrafeedback, save_rows


def prepare_ppo(cfg: dict[str, Any], *, skip_decontam: bool = False) -> Path:
    from data_tools import build_ultrafeedback_prompt_pool

    ds = load_ultrafeedback(cfg, default_split="test_prefs")
    ngram_bank = None if skip_decontam else build_eval_decontam_bank()
    num_prompts = int(cfg["num_prompts"])
    seed = int(cfg["seed"])
    print(f"building ppo prompts num_prompts={num_prompts} seed={seed}")
    subset = build_ultrafeedback_prompt_pool(
        ds, num_prompts=num_prompts, seed=seed, ngram_bank=ngram_bank
    )
    return save_rows(subset, cfg["processed_path"])

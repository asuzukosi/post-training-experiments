"""prepare the stratified tulu sft subset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.prepare.decontam import build_eval_decontam_bank
from pipeline.prepare.io import save_rows


def prepare_sft(cfg: dict[str, Any], *, skip_decontam: bool = False) -> Path:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from pipeline.data_tools import DEFAULT_TOKENIZER_ID, build_tulu_sft_subset

    dataset_id = cfg["dataset"]
    num_samples = int(cfg["num_samples"])
    seed = int(cfg["seed"])
    max_length = int(cfg.get("max_length", 4096))
    out_path = cfg["processed_path"]

    print(f"loading sft dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train")
    print(f"sft raw rows: {len(ds)}")

    ngram_bank = None if skip_decontam else build_eval_decontam_bank()

    print(f"loading tokenizer: {DEFAULT_TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_ID, trust_remote_code=True)

    print(f"building sft subset num_samples={num_samples} seed={seed}")
    subset = build_tulu_sft_subset(
        ds,
        num_samples=num_samples,
        seed=seed,
        ngram_bank=ngram_bank,
        tokenizer=tokenizer,
        max_length=max_length,
        drop_over_max_length=True,
    )
    return save_rows(subset, out_path)

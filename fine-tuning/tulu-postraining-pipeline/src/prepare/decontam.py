"""build the eval 8-gram decontam bank used by sft prep."""
from __future__ import annotations

import json


def build_eval_decontam_bank() -> set[str]:
    """8-gram bank from eval prompts (excludes mmlu auxiliary_train)."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    from data_tools import build_ngram_bank

    texts: list[str] = []

    print("loading decontam texts: reward-bench filtered")
    rb = load_dataset("allenai/reward-bench", split="filtered")
    texts.extend(rb["prompt"])

    print("loading decontam texts: alpaca_eval")
    alpaca_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(alpaca_path) as f:
        alpaca = json.load(f)
    for row in alpaca:
        texts.append(row.get("instruction") or row.get("prompt") or "")

    print("loading decontam texts: ifeval")
    ifeval = load_dataset("google/IFEval", split="train")
    texts.extend(ifeval["prompt"])

    print("loading decontam texts: mmlu test/validation/dev")
    mmlu = load_dataset("cais/mmlu", "all")
    for split in ("test", "validation", "dev"):
        texts.extend(mmlu[split]["question"])

    bank = build_ngram_bank(texts)
    print(f"decontam bank size: {len(bank)} unique 8-grams from {len(texts)} texts")
    return bank

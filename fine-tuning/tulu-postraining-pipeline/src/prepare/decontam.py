"""build the eval 8-gram decontam bank used by sft prep."""
from __future__ import annotations


def build_eval_decontam_bank() -> set[str]:
    """8-gram bank from the eval sets we actually score on.

    excludes mmlu auxiliary_train. only benchmarks we report against belong here —
    decontaminating against one we do not score on drops training rows for nothing.
    """
    from datasets import load_dataset

    from data_tools import build_ngram_bank

    texts: list[str] = []

    print("loading decontam texts: reward-bench filtered")
    rb = load_dataset("allenai/reward-bench", split="filtered")
    texts.extend(rb["prompt"])

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

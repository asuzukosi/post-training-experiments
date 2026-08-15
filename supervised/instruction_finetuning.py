"""explore instruction-tuned data formats and compare base vs finetuned pythia."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jsonlines
import torch
from datasets import load_dataset
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def process_alpaca(n: int) -> list[dict]:
    stream = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    rows = list(itertools.islice(stream, n))
    print("instruction-tuned dataset samples:")
    for row in rows:
        print(row)

    processed = []
    for row in rows:
        if not row["input"]:
            prompt = PROMPT_WITHOUT_INPUT.format(instruction=row["instruction"])
        else:
            prompt = PROMPT_WITH_INPUT.format(
                instruction=row["instruction"],
                input=row["input"],
            )
        processed.append({"input": prompt, "output": row["output"]})
    return processed


def inference(
    text: str,
    model,
    tokenizer,
    max_input_tokens: int = 1000,
    max_output_tokens: int = 100,
) -> str:
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    device = model.device
    generated = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens,
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return decoded[len(text) :]


def compare_models(base_name: str, finetuned_name: str, dataset_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base = AutoModelForCausalLM.from_pretrained(base_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)

    ds = load_dataset(dataset_name)
    sample = ds["test"][0]
    print(sample)
    print("base:", inference(sample["question"], base, tokenizer))

    tuned = AutoModelForCausalLM.from_pretrained(finetuned_name)
    tuned.to(device)
    print("finetuned:", inference(sample["question"], tuned, tokenizer))


def main() -> None:
    parser = argparse.ArgumentParser(description="instruction finetuning exploration")
    parser.add_argument("--alpaca-n", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("alpaca_processed.jsonl"))
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--base-model", default="EleutherAI/pythia-70m")
    parser.add_argument("--finetuned-model", default="lamini/lamini_docs_finetuned")
    parser.add_argument("--dataset", default="lamini/lamini_docs")
    args = parser.parse_args()

    processed = process_alpaca(args.alpaca_n)
    pprint(processed[0])
    with jsonlines.open(args.out, "w") as writer:
        writer.write_all(processed)
    print(f"wrote {len(processed)} rows to {args.out}")

    if not args.skip_compare:
        compare_models(args.base_model, args.finetuned_model, args.dataset)


if __name__ == "__main__":
    main()

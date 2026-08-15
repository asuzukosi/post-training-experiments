"""tokenize instruction data and build train/test splits for supervised finetuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import pandas as pd
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "EleutherAI/pythia-70m"
PROMPT_TEMPLATE = """### Question:
{question}

### Answer:"""


def demo_tokenizer(tokenizer_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    text = "Hi, how are you?"
    encoded = tokenizer(text)["input_ids"]
    print(f"encoded: {encoded}")
    print(f"decoded: {tokenizer.decode(encoded)}")

    list_texts = ["Hi, how are you?", "I'm good", "Yes"]
    print(f"batch: {tokenizer(list_texts)['input_ids']}")
    print(f"padded: {tokenizer(list_texts, padding=True)['input_ids']}")
    print(f"truncated: {tokenizer(list_texts, max_length=3, truncation=True)['input_ids']}")

    tokenizer.truncation_side = "left"
    print(
        f"left truncate + pad: "
        f"{tokenizer(list_texts, max_length=3, truncation=True, padding=True)['input_ids']}"
    )


def load_instruction_examples(path: Path) -> list[dict]:
    df = pd.read_json(path, lines=True)
    examples = df.to_dict()
    rows = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        rows.append(
            {
                "question": PROMPT_TEMPLATE.format(question=question),
                "answer": answer,
            }
        )
    return rows


def make_tokenize_fn(tokenizer):
    def tokenize_function(examples):
        if "question" in examples and "answer" in examples:
            text = examples["question"][0] + examples["answer"][0]
        elif "input" in examples and "output" in examples:
            text = examples["input"][0] + examples["output"][0]
        else:
            text = examples["text"][0]

        tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(text, return_tensors="np", padding=True)
        max_length = min(tokenized["input_ids"].shape[1], 2048)
        tokenizer.truncation_side = "left"
        return tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
        )

    return tokenize_function


def prepare_dataset(
    data_path: Path,
    tokenizer_name: str,
    test_size: float,
    seed: int,
) -> datasets.DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    loaded = datasets.load_dataset("json", data_files=str(data_path), split="train")
    tokenized = loaded.map(
        make_tokenize_fn(tokenizer),
        batched=True,
        batch_size=1,
        drop_last_batch=True,
    )
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    return tokenized.train_test_split(test_size=test_size, shuffle=True, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare supervised finetuning data")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--data", type=Path, default=Path("lamini_docs.jsonl"))
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--demo-only", action="store_true", help="only run tokenizer demos")
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="optional hf dataset id to print (e.g. lamini/lamini_docs)",
    )
    args = parser.parse_args()

    demo_tokenizer(args.tokenizer)
    if args.demo_only:
        return

    if args.data.exists():
        examples = load_instruction_examples(args.data)
        print(f"loaded {len(examples)} examples from {args.data}")
        print(f"sample: {examples[0]}")
        split = prepare_dataset(args.data, args.tokenizer, args.test_size, args.seed)
        print(split)
    else:
        print(f"local file not found: {args.data} (skipping tokenize/split)")

    if args.hf_dataset:
        ds = datasets.load_dataset(args.hf_dataset)
        print(ds)


if __name__ == "__main__":
    main()

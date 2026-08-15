"""compare pretraining-style text with instruction finetuning formats."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jsonlines
import pandas as pd
from datasets import load_dataset
from pprint import pprint

PROMPT_QA = """### Question:
{question}

### Answer:
{answer}"""

PROMPT_Q = """### Question:
{question}

### Answer:"""


def peek_pretrain(n: int) -> None:
    pretrained = load_dataset("c4", "en", split="train", streaming=True)
    print("pretrained dataset samples:")
    for row in itertools.islice(pretrained, n):
        print(row)


def format_instruction_data(path: Path) -> tuple[list[dict], list[dict]]:
    df = pd.read_json(path, lines=True)
    examples = df.to_dict()
    text_only = []
    question_answer = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        text_only.append({"text": PROMPT_QA.format(question=question, answer=answer)})
        question_answer.append(
            {
                "question": PROMPT_Q.format(question=question),
                "answer": answer,
            }
        )
    return text_only, question_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="format finetuning data from jsonl")
    parser.add_argument("--data", type=Path, default=Path("lamini_docs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("lamini_docs_processed.jsonl"))
    parser.add_argument("--peek-pretrain", type=int, default=0, help="n c4 samples to print")
    parser.add_argument("--hf-dataset", default="lamini/lamini_docs")
    args = parser.parse_args()

    if args.peek_pretrain > 0:
        peek_pretrain(args.peek_pretrain)

    if not args.data.exists():
        print(f"local file not found: {args.data}; loading {args.hf_dataset} instead")
        ds = load_dataset(args.hf_dataset)
        print(ds)
        return

    text_only, question_answer = format_instruction_data(args.data)
    pprint(text_only[0])
    pprint(question_answer[0])

    with jsonlines.open(args.out, "w") as writer:
        writer.write_all(question_answer)
    print(f"wrote {len(question_answer)} rows to {args.out}")

    ds = load_dataset(args.hf_dataset)
    print(ds)


if __name__ == "__main__":
    main()

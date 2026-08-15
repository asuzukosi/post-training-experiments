"""evaluate a causal lm on question/answer pairs with exact-match scoring."""

from __future__ import annotations

import argparse

import datasets
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def inference(
    text: str,
    model,
    tokenizer,
    max_input_tokens: int = 1000,
    max_output_tokens: int = 100,
) -> str:
    tokenizer.pad_token = tokenizer.eos_token
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


def evaluate(
    model_name: str,
    dataset_name: str,
    limit: int,
) -> pd.DataFrame:
    dataset = datasets.load_dataset(dataset_name)
    test_dataset = dataset["test"]
    print(test_dataset[0]["question"])
    print(test_dataset[0]["answer"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics = {"exact_matches": []}
    predictions = []
    for i, item in tqdm(enumerate(test_dataset)):
        question = item["question"]
        answer = item["answer"]
        try:
            predicted = inference(question, model, tokenizer)
        except Exception as exc:
            print(f"skip i={i}: {exc}")
            continue
        predictions.append([predicted, answer])
        metrics["exact_matches"].append(is_exact_match(predicted, answer))
        if limit != -1 and i >= limit:
            break

    print(f"number of exact matches: {sum(metrics['exact_matches'])}")
    return pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate supervised finetuned model")
    parser.add_argument("--model", default="lamini/lamini_docs_finetuned")
    parser.add_argument("--dataset", default="lamini/lamini_docs")
    parser.add_argument("--limit", type=int, default=10, help="-1 for full test set")
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help="optional prepared eval set (e.g. lamini/lamini_docs_evaluation)",
    )
    args = parser.parse_args()

    df = evaluate(args.model, args.dataset, args.limit)
    print(df)

    if args.eval_dataset:
        evaluation_dataset = datasets.load_dataset(args.eval_dataset)
        print(pd.DataFrame(evaluation_dataset))


if __name__ == "__main__":
    main()

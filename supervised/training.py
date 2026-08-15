"""short supervised finetune of a small causal lm on lamini docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROMPT_TEMPLATE = """### Question:
{question}

### Answer:"""


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


def tokenize_and_split(
    dataset_path: str,
    use_hf: bool,
    tokenizer,
    max_length: int,
    test_size: float,
    seed: int,
):
    if use_hf:
        loaded = datasets.load_dataset(dataset_path)
        if "train" in loaded:
            raw = loaded["train"]
        else:
            raw = loaded[list(loaded.keys())[0]]
    else:
        raw = datasets.load_dataset("json", data_files=dataset_path, split="train")

    def _format(example):
        question = example.get("question", example.get("input", ""))
        answer = example.get("answer", example.get("output", ""))
        prompt = PROMPT_TEMPLATE.format(question=question)
        text = prompt + answer
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        tokens["question"] = question
        tokens["answer"] = answer
        return tokens

    tokenized = raw.map(_format, remove_columns=raw.column_names)
    return tokenized.train_test_split(test_size=test_size, shuffle=True, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="train a small supervised finetune")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", default="lamini/lamini_docs")
    parser.add_argument("--local-jsonl", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--compare-finetuned",
        default="lamini/lamini_docs_finetuned",
        help="optional longer-trained model to compare after local steps",
    )
    args = parser.parse_args()

    use_hf = args.local_jsonl is None
    dataset_path = args.dataset if use_hf else str(args.local_jsonl)
    output_dir = args.output_dir or Path(f"lamini_docs_{args.max_steps}_steps")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    split = tokenize_and_split(
        dataset_path,
        use_hf,
        tokenizer,
        args.max_length,
        args.test_size,
        args.seed,
    )
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(train_dataset)
    print(test_dataset)

    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    test_text = test_dataset[0]["question"]
    print("question input (test):", test_text)
    print(f"correct answer: {test_dataset[0]['answer']}")
    print("base model answer:", inference(test_text, base_model, tokenizer))

    training_args = TrainingArguments(
        learning_rate=1.0e-5,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        disable_tqdm=False,
        eval_steps=120,
        save_steps=120,
        warmup_steps=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        optim="adafactor",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    print(base_model)
    print("memory footprint", base_model.get_memory_footprint() / 1e9, "gb")

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    save_dir = output_dir / "final"
    trainer.save_model(str(save_dir))
    print("saved model to:", save_dir)

    slightly = AutoModelForCausalLM.from_pretrained(str(save_dir), local_files_only=True)
    slightly.to(device)
    print("finetuned slightly:", inference(test_text, slightly, tokenizer))
    print("target answer:", test_dataset[0]["answer"])

    if args.compare_finetuned:
        longer = AutoModelForCausalLM.from_pretrained(args.compare_finetuned)
        longer_tok = AutoTokenizer.from_pretrained(args.compare_finetuned)
        longer.to(device)
        print("finetuned longer:", inference(test_text, longer, longer_tok))
        print("mars (base):", inference("What do you think of Mars?", slightly, tokenizer))
        print(
            "mars (longer):",
            inference("What do you think of Mars?", longer, longer_tok),
        )


if __name__ == "__main__":
    main()

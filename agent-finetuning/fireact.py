"""fireact-style agent finetuning helpers.

openai path: upload jsonl trajectories and manage fine-tuning jobs.
llama path: full-model hf trainer scaffold (requires local data + gpu).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import openai
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

DEFAULT_GPT_BASE = "gpt-3.5-turbo"
DEFAULT_DATA = Path(__file__).resolve().parent / "data.jsonl"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_client() -> openai.OpenAI:
    return openai.OpenAI()


def list_files(client: openai.OpenAI) -> None:
    files = client.files.list()
    rows = sorted(files.data, key=lambda item: -item.created_at)
    print(pd.DataFrame([item.model_dump() for item in rows]).to_string(index=False))


def list_jobs(client: openai.OpenAI) -> None:
    jobs = client.fine_tuning.jobs.list()
    print(pd.DataFrame([item.model_dump() for item in jobs.data]).to_string(index=False))


def list_models(client: openai.OpenAI) -> None:
    models = client.models.list()
    print(pd.DataFrame([item.model_dump() for item in models.data]).to_string(index=False))


def upload_file(client: openai.OpenAI, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.files.create(file=handle, purpose="fine-tune")
    print(f"uploaded file_id={response.id}")
    return response.id


def create_job(client: openai.OpenAI, file_id: str, model: str, n_epochs: int) -> str:
    job = client.fine_tuning.jobs.create(
        model=model,
        training_file=file_id,
        hyperparameters={"n_epochs": n_epochs},
    )
    print(f"created job_id={job.id} status={job.status}")
    return job.id


def retrieve_job(client: openai.OpenAI, job_id: str) -> None:
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(json.dumps(job.model_dump(), indent=2, default=str))


def list_events(client: openai.OpenAI, job_id: str) -> None:
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
    for event in events.data:
        print(event.created_at, event.message)


def cancel_job(client: openai.OpenAI, job_id: str) -> None:
    job = client.fine_tuning.jobs.cancel(job_id)
    print(f"cancelled job_id={job.id} status={job.status}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "path to training data json/jsonl"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "max sequence length; sequences are right-padded or truncated"},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens <= 0:
        return
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_avg
    output_embeddings[-num_new_tokens:] = output_avg


class SupervisedDataset(Dataset):
    """simple instruction dataset for fireact-style trajectories."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"training data not found: {path}")

        examples = []
        if path.suffix == ".jsonl":
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        examples.append(json.loads(line))
        else:
            examples = json.loads(path.read_text())

        sources, targets = [], []
        for example in examples:
            if "messages" in example:
                # openai chat-style row → flatten to prompt/response
                user_parts = [m["content"] for m in example["messages"] if m["role"] == "user"]
                assistant_parts = [
                    m["content"] for m in example["messages"] if m["role"] == "assistant"
                ]
                instruction = user_parts[0] if user_parts else ""
                output = assistant_parts[-1] if assistant_parts else ""
                source = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
            else:
                instruction = example.get("instruction", "")
                inp = example.get("input", "")
                output = example.get("output", example.get("response", ""))
                if inp:
                    source = PROMPT_DICT["prompt_input"].format(
                        instruction=instruction, input=inp
                    )
                else:
                    source = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
            sources.append(source)
            targets.append(f"{output}{tokenizer.eos_token}")

        tokenized_sources = tokenizer(
            sources,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        tokenized_targets = tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = torch.cat(
            [tokenized_sources.input_ids, tokenized_targets.input_ids], dim=1
        )
        labels = input_ids.clone()
        source_lens = tokenized_sources.attention_mask.sum(dim=1)
        for i, source_len in enumerate(source_lens):
            labels[i, : int(source_len)] = IGNORE_INDEX
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
) -> dict:
    train_dataset = SupervisedDataset(data_path=data_args.data_path, tokenizer=tokenizer)
    return {"train_dataset": train_dataset}


def train_llama() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=False
    )
    if not data_args.data_path:
        raise SystemExit("--data_path is required for llama training")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if "code" in model_args.model_name_or_path.lower():
        tokenizer.add_eos_token = True
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN},
            tokenizer=tokenizer,
            model=model,
        )

    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fireact agent finetuning helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-files", help="list openai files")
    sub.add_parser("list-jobs", help="list fine-tuning jobs")
    sub.add_parser("list-models", help="list available models")

    upload = sub.add_parser("upload", help="upload a jsonl training file")
    upload.add_argument("--data", type=Path, default=DEFAULT_DATA)

    create = sub.add_parser("create-job", help="create a fine-tuning job")
    create.add_argument("--file-id", required=True)
    create.add_argument("--model", default=DEFAULT_GPT_BASE)
    create.add_argument("--n-epochs", type=int, default=20)

    retrieve = sub.add_parser("retrieve", help="retrieve a job")
    retrieve.add_argument("--job-id", required=True)

    events = sub.add_parser("events", help="list job events")
    events.add_argument("--job-id", required=True)

    cancel = sub.add_parser("cancel", help="cancel a job")
    cancel.add_argument("--job-id", required=True)

    sub.add_parser(
        "llama-train",
        help="run llama/opt full finetune via transformers (pass hf args after --)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    # allow: python fireact.py llama-train --model_name_or_path ... --data_path ...
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "llama-train":
        sys_argv_backup = sys.argv[:]
        sys.argv = [sys_argv_backup[0], *argv[1:]]
        try:
            train_llama()
        finally:
            sys.argv = sys_argv_backup
        return

    parser = build_parser()
    args = parser.parse_args(argv)
    client = get_client()

    if args.command == "list-files":
        list_files(client)
    elif args.command == "list-jobs":
        list_jobs(client)
    elif args.command == "list-models":
        list_models(client)
    elif args.command == "upload":
        upload_file(client, args.data)
    elif args.command == "create-job":
        create_job(client, args.file_id, args.model, args.n_epochs)
    elif args.command == "retrieve":
        retrieve_job(client, args.job_id)
    elif args.command == "events":
        list_events(client, args.job_id)
    elif args.command == "cancel":
        cancel_job(client, args.job_id)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

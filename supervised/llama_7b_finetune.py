"""qlora finetune of llama-2 7b on alpaca-gpt4 (colab-oriented scaffold)."""

from __future__ import annotations

import argparse
import os
import platform

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    TrainingArguments,
)
from trl import SFTTrainer


def print_system_specs() -> None:
    is_cuda_available = torch.cuda.is_available()
    print("cuda available:", is_cuda_available)
    num_cuda_devices = torch.cuda.device_count()
    print("number of cuda devices:", num_cuda_devices)
    if is_cuda_available:
        for i in range(num_cuda_devices):
            print(f"--- cuda devices {i} ---")
            print("name:", torch.cuda.get_device_name(i))
            print("compute capability:", torch.cuda.get_device_capability(i))
            print(
                "total memory:",
                torch.cuda.get_device_properties(i).total_memory,
                "bytes",
            )
    print("--cpu information--")
    print("processor:", platform.processor())
    print("system:", platform.system(), platform.release())
    print("python version:", platform.python_version())


def stream(model, tokenizer, user_prompt: str, device: str, max_new_tokens: int) -> None:
    system_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. \n\n"
    )
    b_inst, e_inst = "### Instruction: \n", "### Response: \n"
    prompt = f"{system_prompt}{b_inst}{user_prompt.strip()} \n\n {e_inst}"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)


def train(args: argparse.Namespace) -> None:
    print_system_specs()
    if args.hf_token:
        login(token=args.hf_token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    dataset = load_dataset(args.dataset, split=args.dataset_split)
    print(dataset["output"][0])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    if args.wandb:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            job_type="training",
            anonymous="allow",
        )

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
        ],
    )
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=30,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.3,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="wandb" if args.wandb else "none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=None,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    trainer.train()
    trainer.model.save_pretrained(args.adapter_dir)
    if args.wandb:
        wandb.finish()
    model.config.use_cache = True
    model.eval()

    stream(model, tokenizer, args.prompt, "cuda:0", args.max_new_tokens)

    if args.merge_and_push:
        del model, trainer
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        merged = PeftModel.from_pretrained(base_model, args.adapter_dir)
        merged = merged.merge_and_unload()
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        merged.push_to_hub(args.push_name)
        tok.push_to_hub(args.push_name)


def chat(args: argparse.Namespace) -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stream(model, tokenizer, args.prompt, "cuda:0", args.max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="llama-2 7b qlora alpaca finetune")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="run qlora training")
    train_p.add_argument("--model", default="georgesung/llama2_7b_chat_uncensored")
    train_p.add_argument("--dataset", default="vicgalle/alpaca-gpt4")
    train_p.add_argument("--dataset-split", default="train[:10000]")
    train_p.add_argument("--adapter-dir", default="kosiasuzu/llama2-7b-aplaca-finetune")
    train_p.add_argument("--output-dir", default="./results")
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--prompt", default="what is newtons 3rd law and its formula")
    train_p.add_argument("--max-new-tokens", type=int, default=500)
    train_p.add_argument("--hf-token", default=None)
    train_p.add_argument("--wandb", action="store_true")
    train_p.add_argument("--wandb-project", default="fine tuning llama-2-7b")
    train_p.add_argument("--merge-and-push", action="store_true")
    train_p.add_argument(
        "--push-name",
        default="kosiasuzu/llama2-7b-aplaca-finetune2",
    )

    chat_p = sub.add_parser("chat", help="generate from a hub model")
    chat_p.add_argument("--model", default="kosiasuzu/llama2-7b-alpaca-finetune")
    chat_p.add_argument("--prompt", default="what is newtons 3rd law and its formula")
    chat_p.add_argument("--max-new-tokens", type=int, default=500)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        chat(args)


if __name__ == "__main__":
    main()

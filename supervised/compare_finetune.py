"""compare base llama-2 vs chat (instruction-tuned) completions via transformers."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Tell me how to train my dog to sit",
    "What do you think of Mars?",
    "taylor swift's best friend",
    (
        "Agent: I'm here to help you with your Amazon deliver order.\n"
        "Customer: I didn't get my item\n"
        "Agent: I'm sorry to hear that. Which item was it?\n"
        "Customer: the blanket\n"
        "Agent:"
    ),
]


def load_runner(model_name: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def run(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="compare base vs chat llama-2")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--chat-model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    if not args.skip_base:
        print("=== non-finetuned / base ===")
        base = load_runner(args.base_model, args.max_new_tokens)
        for prompt in PROMPTS:
            print(f"\nprompt: {prompt[:80]}...")
            print(base(prompt))
        print("\nwith inst tags (base):")
        print(base("[INST]Tell me how to train my dog to sit[/INST]"))

    if not args.skip_chat:
        print("\n=== chat / instruction-tuned ===")
        chat = load_runner(args.chat_model, args.max_new_tokens)
        for prompt in PROMPTS:
            print(f"\nprompt: {prompt[:80]}...")
            print(chat(prompt))
        print("\nwith inst tags (chat):")
        print(chat("[INST]Tell me how to train my dog to sit[/INST]"))


if __name__ == "__main__":
    main()

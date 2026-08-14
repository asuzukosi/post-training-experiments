"""ppo prompt-pool prep: load, render + tokenize, and hold out a sampling set."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.paths import resolve_path

# trl samples completions at this cadence by default (PPOConfig.num_sample_generations)
DEFAULT_NUM_SAMPLE_GENERATIONS = 10
# prompts held out purely to feed those samples; never trained on
DEFAULT_EVAL_PROMPTS = 32
# 1 keeps the split valid at any size — see split_ppo_eval on trl's drop_last
DEFAULT_EVAL_BATCH_SIZE = 1


def ppo_eval_batch_size(cfg: dict[str, Any]) -> int:
    """single source of truth for the eval batch size (config and split must agree)."""
    return int(cfg.get("per_device_eval_batch_size", DEFAULT_EVAL_BATCH_SIZE))


def load_ppo_prompts(processed_path: str | Path):
    """load prepared ppo prompt pool (expects `prompt`)."""
    from datasets import load_from_disk

    path = resolve_path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ppo processed dataset not found: {path}; "
            "run: python scripts/prepare/ppo.py"
        )
    ds = load_from_disk(str(path))
    if "prompt" not in ds.column_names:
        raise ValueError(f"ppo dataset at {path} missing 'prompt' column")
    print(f"loaded ppo prompts: {path} rows={len(ds)}")
    return ds


def tokenize_ppo_prompts(dataset, tokenizer: Any, *, max_prompt_length: int):
    """apply chat template + tokenize prompts to `input_ids` for PPOTrainer."""

    def _render(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _tokenize(batch):
        texts = [_render(p) for p in batch["prompt"]]
        out = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )
        return {"input_ids": out["input_ids"]}

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"tokenized ppo prompts: rows={len(tokenized)} max_prompt_length={max_prompt_length}")
    return tokenized


def split_ppo_eval(
    dataset,
    *,
    num_eval: int = DEFAULT_EVAL_PROMPTS,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
):
    """split off prompts for trl's periodic completion sampling.

    ppo REQUIRES this in a way the other stages do not, and gets it wrong twice:

    1. trl 0.19's PPOTrainer.train() calls generate_completions() whenever
       num_sample_generations > 0 (default 10), and that path iterates
       self.eval_dataloader. with no eval_dataset it dies on
       `object of type 'NoneType' has no len()`.
    2. that dataloader is built with drop_last=True, so an eval split smaller than
       eval_batch_size is dropped whole, the iteration yields None, and the next line
       dies on `'NoneType' object is not subscriptable`.

    both fire *after* training has started — i.e. deep into an expensive run — and
    sft/rm/dpo all treat eval_dataset as optional, so nothing upstream catches either.

    the sampled completions are also the cheapest way to see reward hacking as it starts,
    so hold prompts out rather than setting num_sample_generations to 0.

    returns (train, eval); eval is None only when the pool cannot spare a whole batch.
    """
    if num_eval <= 0 or eval_batch_size <= 0 or len(dataset) < 2:
        return dataset, None
    # never surrender more than a quarter of a small pool to sampling...
    num_eval = min(num_eval, max(1, len(dataset) // 4))
    # ...then keep whole batches only, or drop_last discards the lot
    num_eval = (num_eval // eval_batch_size) * eval_batch_size
    if num_eval < eval_batch_size or num_eval >= len(dataset):
        print(
            f"ppo prompt split: pool of {len(dataset)} cannot spare a whole eval batch "
            f"of {eval_batch_size}; completion sampling disabled"
        )
        return dataset, None
    eval_ds = dataset.select(range(num_eval))
    train_ds = dataset.select(range(num_eval, len(dataset)))
    print(f"ppo prompt split: train={len(train_ds)} eval={len(eval_ds)} (completion sampling)")
    return train_ds, eval_ds

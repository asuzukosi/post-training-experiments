"""ppo training: align sft policy against rm, kl/clip/eos, step ckpts, hub push.

trl 0.19 PPOTrainer.train() does not accept resume_from_checkpoint; if a step
checkpoint exists under the run dir we re-init the policy from it (weights only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_tools.chat import ensure_pad_token
from data_tools.naming import checkpoint_dir, make_run_name
from hub import hub_trainer_kwargs, push_checkpoint_to_hub
from prepare.paths import ROOT, resolve_path
from resume import (
    DEFAULT_WANDB_RESUME,
    cfg_use_wandb,
    find_latest_checkpoint,
    trainer_report_to,
    wandb_run,
)

DEFAULT_WANDB_PROJECT = "tulu-postraining"
# used when score_eos_only is true and missing_eos_penalty is unset
DEFAULT_MISSING_EOS_PENALTY = 1.0


def resolve_sft_checkpoint(cfg: dict[str, Any], sft_checkpoint: str | Path | None) -> Path:
    """require an sft checkpoint path (cli override or cfg.sft_checkpoint)."""
    raw = sft_checkpoint if sft_checkpoint is not None else cfg.get("sft_checkpoint")
    if not raw:
        raise ValueError(
            "sft_checkpoint is required to init ppo policy/ref; "
            "pass --sft-checkpoint or set sft_checkpoint in configs/ppo.yaml"
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"sft checkpoint not found: {path}")
    return path


def resolve_rm_checkpoint(cfg: dict[str, Any], rm_checkpoint: str | Path | None) -> Path:
    """require an rm checkpoint path (cli override or cfg.rm_checkpoint)."""
    raw = rm_checkpoint if rm_checkpoint is not None else cfg.get("rm_checkpoint")
    if not raw:
        raise ValueError(
            "rm_checkpoint is required for ppo reward scoring; "
            "pass --rm-checkpoint or set rm_checkpoint in configs/ppo.yaml"
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"rm checkpoint not found: {path}")
    return path


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


def build_ppo_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    output_dir: str | Path,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> Any:
    """build trl PPOConfig from yaml (maps batch_size / ppo_epochs aliases)."""
    from trl import PPOConfig

    out = Path(output_dir)
    if push_to_hub:
        hub_kwargs = hub_trainer_kwargs(
            run_name,
            username=hub_username,
            push_to_hub=True,
        )
    else:
        hub_kwargs = {"push_to_hub": False}

    # yaml uses batch_size; trl 0.19 uses per_device_train_batch_size
    per_device_bs = int(
        cfg.get("per_device_train_batch_size", cfg.get("batch_size", 8))
    )
    num_ppo_epochs = int(cfg.get("num_ppo_epochs", cfg.get("ppo_epochs", 1)))
    response_length = int(cfg.get("response_length", cfg.get("max_new_tokens", 512)))

    # score_eos_only -> missing_eos_penalty (trl subtracts penalty when eos missing)
    if cfg.get("missing_eos_penalty") is not None:
        missing_eos_penalty = float(cfg["missing_eos_penalty"])
    elif bool(cfg.get("score_eos_only", True)):
        missing_eos_penalty = float(DEFAULT_MISSING_EOS_PENALTY)
    else:
        missing_eos_penalty = None

    # mini_batch_size -> num_mini_batches when local batch is known
    num_mini_batches = int(cfg.get("num_mini_batches", 1))
    if "mini_batch_size" in cfg and "num_mini_batches" not in cfg:
        mini = int(cfg["mini_batch_size"])
        grad_acc = int(cfg.get("gradient_accumulation_steps", 1))
        local_bs = per_device_bs * grad_acc
        if mini > 0 and local_bs % mini == 0:
            num_mini_batches = local_bs // mini
        else:
            print(
                f"warning: cannot derive num_mini_batches from "
                f"mini_batch_size={mini} local_bs={local_bs}; using 1"
            )
            num_mini_batches = 1

    return PPOConfig(
        output_dir=str(out),
        run_name=run_name,
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        total_episodes=int(cfg["total_episodes"]) if cfg.get("total_episodes") is not None else None,
        max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else -1,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        num_mini_batches=num_mini_batches,
        num_ppo_epochs=num_ppo_epochs,
        kl_coef=float(cfg.get("kl_coef", 0.05)),
        cliprange=float(cfg.get("cliprange", 0.2)),
        response_length=response_length,
        stop_token="eos",
        missing_eos_penalty=missing_eos_penalty,
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        report_to=trainer_report_to(cfg_use_wandb(cfg)),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_steps=int(cfg.get("save_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        remove_unused_columns=False,
        **hub_kwargs,
    )


def build_ppo_trainer(
    cfg: dict[str, Any],
    *,
    sft_checkpoint: str | Path | None = None,
    rm_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> tuple[Any, str, Path]:
    """construct PPOTrainer; returns (trainer, run_name, output_dir)."""
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl import PPOTrainer

    sft_path = resolve_sft_checkpoint(cfg, sft_checkpoint)
    rm_path = resolve_rm_checkpoint(cfg, rm_checkpoint)
    base_name = cfg["base_name"]
    task = cfg.get("task", "ppo")
    run = run_name or make_run_name(base_name, task)
    out = Path(output_dir) if output_dir is not None else checkpoint_dir(run)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    # weights-only resume: continue policy from latest step ckpt if present
    policy_path = sft_path
    latest = find_latest_checkpoint(out)
    if latest is not None:
        print(f"ppo weights resume from checkpoint: {latest}")
        policy_path = latest

    train_raw = dataset if dataset is not None else load_ppo_prompts(cfg["processed_path"])
    max_prompt_length = int(cfg.get("max_prompt_length", 1024))

    print(f"loading tokenizer from sft: {sft_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)
    tokenizer = ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"

    train_ds = tokenize_ppo_prompts(
        train_raw,
        tokenizer,
        max_prompt_length=max_prompt_length,
    )

    print(f"loading policy from: {policy_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        str(policy_path),
        trust_remote_code=True,
        torch_dtype="auto",
    )
    print(f"loading ref policy from sft: {sft_path}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        str(sft_path),
        trust_remote_code=True,
        torch_dtype="auto",
    )
    print(f"loading reward model from: {rm_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(rm_path),
        num_labels=1,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    reward_model.requires_grad_(False)
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    # critic: scalar head on sft backbone (fresh value head)
    print(f"loading value model from sft: {sft_path}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        str(sft_path),
        num_labels=1,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    args = build_ppo_config(
        cfg,
        run_name=run,
        output_dir=out,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_ds,
    )
    return trainer, run, out


def run_ppo(
    cfg: dict[str, Any],
    *,
    sft_checkpoint: str | Path | None = None,
    rm_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
    wandb_project: str | None = None,
) -> Path:
    """train ppo end-to-end: wandb, train, save, optional hub push."""

    trainer, run, out = build_ppo_trainer(
        cfg,
        sft_checkpoint=sft_checkpoint,
        rm_checkpoint=rm_checkpoint,
        run_name=run_name,
        output_dir=output_dir,
        dataset=dataset,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    project = wandb_project or cfg.get("wandb_project") or DEFAULT_WANDB_PROJECT
    with wandb_run(
        use_wandb=cfg_use_wandb(cfg),
        output_dir=out,
        project=project,
        name=run,
        resume=cfg.get("wandb_resume", DEFAULT_WANDB_RESUME),
    ):
        print(f"starting ppo run_name={run} output_dir={out}")
        trainer.train()
        trainer.save_model(str(out))
        # writes trainer_state.json (log_history: loss, grad_norm per step) next to the
        # model — makes a run auditable, and lets a smoke prove training actually happened
        trainer.save_state()
        trainer.processing_class.save_pretrained(str(out))

        if push_to_hub:
            push_checkpoint_to_hub(out, run_name=run, username=hub_username)

    print(f"ppo done: {out}")
    return out

"""PPOConfig / PPOTrainer construction and the ppo run entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_tools.chat import ensure_pad_token
from data_tools.naming import checkpoint_dir, make_run_name
from hub import hub_trainer_kwargs, push_checkpoint_to_hub
from prepare.paths import ROOT
from resume import (
    DEFAULT_WANDB_RESUME,
    cfg_use_wandb,
    find_latest_checkpoint,
    trainer_report_to,
    wandb_run,
)
from trainers.tripwire import attach_if_configured
from trainers.ppo.data import (
    DEFAULT_EVAL_PROMPTS,
    DEFAULT_NUM_SAMPLE_GENERATIONS,
    load_ppo_prompts,
    ppo_eval_batch_size,
    split_ppo_eval,
    tokenize_ppo_prompts,
)
from trainers.ppo.models import (
    load_ppo_models,
    resolve_rm_checkpoint,
    resolve_sft_checkpoint,
)

DEFAULT_WANDB_PROJECT = "tulu-postraining"
# used when score_eos_only is true and missing_eos_penalty is unset
DEFAULT_MISSING_EOS_PENALTY = 1.0


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
        # must match split_ppo_eval's: trl drops the last partial eval batch
        per_device_eval_batch_size=ppo_eval_batch_size(cfg),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        num_mini_batches=num_mini_batches,
        num_ppo_epochs=num_ppo_epochs,
        kl_coef=float(cfg.get("kl_coef", 0.05)),
        cliprange=float(cfg.get("cliprange", 0.2)),
        response_length=response_length,
        # >0 makes train() sample completions, which REQUIRES an eval_dataset
        num_sample_generations=int(
            cfg.get("num_sample_generations", DEFAULT_NUM_SAMPLE_GENERATIONS)
        ),
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
    from transformers import AutoTokenizer
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

    tokenized = tokenize_ppo_prompts(
        train_raw,
        tokenizer,
        max_prompt_length=max_prompt_length,
    )
    # trl samples completions from eval_dataset during train(); see split_ppo_eval.
    # the batch size must match build_ppo_config's, or drop_last empties the split.
    train_ds, eval_ds = split_ppo_eval(
        tokenized,
        num_eval=int(cfg.get("num_eval_prompts", DEFAULT_EVAL_PROMPTS)),
        eval_batch_size=ppo_eval_batch_size(cfg),
    )

    policy, ref_policy, reward_model, value_model = load_ppo_models(
        sft_path=sft_path,
        rm_path=rm_path,
        policy_path=policy_path,
        pad_token_id=tokenizer.pad_token_id,
    )

    args = build_ppo_config(
        cfg,
        run_name=run,
        output_dir=out,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )
    # a pool too small to spare a whole eval batch leaves eval_ds None; sampling would
    # then crash in generate_completions, so turn it off rather than ship a broken run
    if eval_ds is None and args.num_sample_generations:
        print("no ppo eval split available; disabling completion sampling")
        args.num_sample_generations = 0

    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    attach_if_configured(trainer, cfg, tokenizer)
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

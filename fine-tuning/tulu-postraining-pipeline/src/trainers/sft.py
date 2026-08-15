"""sft training: packing, assistant-only loss, resume, wandb, hub push."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from trainers.tripwire import attach_if_configured
from data_tools.chat import ensure_assistant_generation_template, ensure_pad_token
from data_tools.naming import checkpoint_dir, make_run_name
from hub import hub_trainer_kwargs, push_checkpoint_to_hub
from prepare.paths import ROOT, resolve_path
from resume import (
    DEFAULT_WANDB_RESUME,
    cfg_use_wandb,
    resolve_resume_from_checkpoint,
    trainer_report_to,
    wandb_run,
)
DEFAULT_WANDB_PROJECT = "tulu-postraining"


def resolve_sft_checkpoint(
    cfg: dict[str, Any],
    sft_checkpoint: str | Path | None = None,
) -> Path:
    """require an sft checkpoint (cli override or cfg.sft_checkpoint)."""
    raw = sft_checkpoint if sft_checkpoint is not None else cfg.get("sft_checkpoint")
    if not raw:
        raise ValueError(
            "sft_checkpoint is required to init from model_sft; "
            "pass --sft-checkpoint or set sft_checkpoint in config"
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"sft checkpoint not found: {path}")
    return path


def load_sft_dataset(processed_path: str | Path):
    """load prepared sft subset from disk (expects conversational `messages`)."""
    from datasets import load_from_disk

    path = resolve_path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"sft processed dataset not found: {path}"
        )
    ds = load_from_disk(str(path))
    if "messages" not in ds.column_names:
        raise ValueError(f"sft dataset at {path} missing 'messages' column")
    print(f"loaded sft dataset: {path} rows={len(ds)}")
    return ds


def build_sft_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    output_dir: str | Path,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> Any:
    """build trl SFTConfig from yaml cfg + run naming / hub kwargs."""
    from trl import SFTConfig

    out = Path(output_dir)
    if push_to_hub:
        hub_kwargs = hub_trainer_kwargs(
            run_name,
            username=hub_username,
            push_to_hub=True,
        )
    else:
        hub_kwargs = {"push_to_hub": False}

    return SFTConfig(
        output_dir=str(out),
        run_name=run_name,
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else -1,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        report_to=trainer_report_to(cfg_use_wandb(cfg)),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        max_length=int(cfg.get("max_length", 4096)),
        packing=bool(cfg.get("packing", True)),
        assistant_only_loss=bool(cfg.get("assistant_only_loss", True)),
        remove_unused_columns=False,
        average_tokens_across_devices=False,
        **hub_kwargs,
    )


def build_sft_trainer(
    cfg: dict[str, Any],
    *,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> tuple[Any, str, Path]:
    """construct SFTTrainer; returns (trainer, run_name, output_dir)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    base_model = cfg["base_model"]
    base_name = cfg["base_name"]
    task = cfg.get("task", "sft")
    run = run_name or make_run_name(base_name, task)
    out = Path(output_dir) if output_dir is not None else checkpoint_dir(run)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    train_ds = dataset if dataset is not None else load_sft_dataset(cfg["processed_path"])

    print(f"loading tokenizer/model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = ensure_pad_token(tokenizer)
    tokenizer = ensure_assistant_generation_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    args = build_sft_config(
        cfg,
        run_name=run,
        output_dir=out,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    attach_if_configured(trainer, cfg, tokenizer)
    return trainer, run, out


def run_sft(
    cfg: dict[str, Any],
    *,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
    wandb_project: str | None = None,
) -> Path:
    """train sft end-to-end: auto-resume, wandb, train, save, optional hub push."""

    trainer, run, out = build_sft_trainer(
        cfg,
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
        resume_ckpt = resolve_resume_from_checkpoint(out, resume=True)
        print(f"starting sft run_name={run} output_dir={out} resume={resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(str(out))
        # writes trainer_state.json (log_history: loss, grad_norm per step) next to the
        # model — makes a run auditable, and lets a smoke prove training actually happened
        trainer.save_state()
        trainer.processing_class.save_pretrained(str(out))

        if push_to_hub:
            push_checkpoint_to_hub(out, run_name=run, username=hub_username)

    print(f"sft done: {out}")
    return out

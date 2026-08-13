"""reward-model training: init from sft, bradley-terry, 1 epoch only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_tools.naming import checkpoint_dir, make_run_name
from hub import hub_trainer_kwargs, push_checkpoint_to_hub
from prepare.paths import ROOT, resolve_path
from resume import (
    DEFAULT_WANDB_RESUME,
    resolve_resume_from_checkpoint,
    wandb_resume_kwargs,
)

DEFAULT_WANDB_PROJECT = "tulu-postraining"
# spec hard rule: rm is 1 epoch only (overfit risk)
RM_NUM_TRAIN_EPOCHS = 1


def load_rm_dataset(processed_path: str | Path):
    """load prepared preference pairs (expects `chosen` / `rejected`)."""
    from datasets import load_from_disk

    path = resolve_path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"rm processed dataset not found: {path}; "
            "run: python scripts/prepare/rm.py"
        )
    ds = load_from_disk(str(path))
    missing = [c for c in ("chosen", "rejected") if c not in ds.column_names]
    if missing:
        raise ValueError(f"rm dataset at {path} missing columns: {missing}")
    print(f"loaded rm dataset: {path} rows={len(ds)}")
    return ds


def resolve_sft_checkpoint(cfg: dict[str, Any], sft_checkpoint: str | Path | None) -> Path:
    """require an sft checkpoint path (cli override or cfg.sft_checkpoint)."""
    raw = sft_checkpoint if sft_checkpoint is not None else cfg.get("sft_checkpoint")
    if not raw:
        raise ValueError(
            "sft_checkpoint is required to init the reward model; "
            "pass --sft-checkpoint or set sft_checkpoint in configs/rm.yaml"
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"sft checkpoint not found: {path}")
    return path


def build_rm_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    output_dir: str | Path,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> Any:
    """build trl RewardConfig; num_train_epochs is always 1."""
    from trl import RewardConfig

    out = Path(output_dir)
    if push_to_hub:
        hub_kwargs = hub_trainer_kwargs(
            run_name,
            username=hub_username,
            push_to_hub=True,
        )
    else:
        hub_kwargs = {"push_to_hub": False}

    cfg_epochs = cfg.get("num_train_epochs")
    if cfg_epochs is not None and float(cfg_epochs) != float(RM_NUM_TRAIN_EPOCHS):
        print(
            f"warning: ignoring num_train_epochs={cfg_epochs}; "
            f"rm is hardcoded to {RM_NUM_TRAIN_EPOCHS} epoch"
        )

    return RewardConfig(
        output_dir=str(out),
        run_name=run_name,
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(RM_NUM_TRAIN_EPOCHS),
        max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else -1,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        report_to=cfg.get("report_to", "wandb"),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        max_length=int(cfg.get("max_length", 2048)),
        remove_unused_columns=False,
        average_tokens_across_devices=False,
        **hub_kwargs,
    )


def build_rm_trainer(
    cfg: dict[str, Any],
    *,
    sft_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> tuple[Any, str, Path]:
    """construct RewardTrainer from sft ckpt; returns (trainer, run_name, output_dir)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardTrainer

    sft_path = resolve_sft_checkpoint(cfg, sft_checkpoint)
    base_name = cfg["base_name"]
    task = cfg.get("task", "rm")
    run = run_name or make_run_name(base_name, task)
    out = Path(output_dir) if output_dir is not None else checkpoint_dir(run)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    train_ds = dataset if dataset is not None else load_rm_dataset(cfg["processed_path"])

    print(f"loading tokenizer/rm head from sft checkpoint: {sft_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # sequence-classification head replaces the causal lm head (scalar reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(sft_path),
        num_labels=1,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    args = build_rm_config(
        cfg,
        run_name=run,
        output_dir=out,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    return trainer, run, out


def run_rm(
    cfg: dict[str, Any],
    *,
    sft_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
    wandb_project: str | None = None,
) -> Path:
    """train rm end-to-end: auto-resume, wandb, train, save, optional hub push."""
    import wandb

    trainer, run, out = build_rm_trainer(
        cfg,
        sft_checkpoint=sft_checkpoint,
        run_name=run_name,
        output_dir=output_dir,
        dataset=dataset,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    report_to = cfg.get("report_to", "wandb")
    use_wandb = report_to not in (None, "", "none", [])
    if use_wandb:
        project = wandb_project or cfg.get("wandb_project") or DEFAULT_WANDB_PROJECT
        wandb_mode = cfg.get("wandb_resume", DEFAULT_WANDB_RESUME)
        wb_kwargs = wandb_resume_kwargs(
            out,
            project=project,
            name=run,
            resume=wandb_mode,
        )
        wandb.init(**wb_kwargs)

    resume_ckpt = resolve_resume_from_checkpoint(out, resume=True)
    print(f"starting rm run_name={run} output_dir={out} resume={resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out))
    trainer.processing_class.save_pretrained(str(out))

    if push_to_hub:
        push_checkpoint_to_hub(out, run_name=run, username=hub_username)

    if use_wandb:
        wandb.finish()
    print(f"rm done: {out}")
    return out

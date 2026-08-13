"""dpo training: one beta arm per run, cached ref log-probs, resume, hub push.

trl DPOTrainer already logs logps/chosen and logps/rejected (displacement signal).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from data_tools.naming import checkpoint_dir, make_run_name
from hub import hub_trainer_kwargs, push_checkpoint_to_hub
from prepare.paths import ROOT, resolve_path
from resume import (
    DEFAULT_WANDB_RESUME,
    resolve_resume_from_checkpoint,
    wandb_resume_kwargs,
)

DEFAULT_WANDB_PROJECT = "tulu-postraining"


def format_dpo_task(beta: float) -> str:
    """build task tag used in run names, e.g. dpo-b0.05 / dpo-b0.1."""
    return f"dpo-b{beta:g}"


def resolve_beta(cfg: dict[str, Any], beta: float | None) -> float:
    """pick one beta arm; must be listed in cfg['betas'] when that list is set."""
    allowed = cfg.get("betas")
    if beta is None:
        if isinstance(allowed, Sequence) and len(allowed) == 1:
            return float(allowed[0])
        raise ValueError(
            "beta is required; pass --beta (one arm per call). "
            f"allowed from config: {allowed}"
        )
    value = float(beta)
    if isinstance(allowed, Sequence) and allowed:
        allowed_f = [float(b) for b in allowed]
        if not any(abs(value - b) < 1e-12 for b in allowed_f):
            raise ValueError(f"beta={value} not in configs betas={allowed_f}")
    return value


def resolve_sft_checkpoint(cfg: dict[str, Any], sft_checkpoint: str | Path | None) -> Path:
    """require an sft checkpoint path (cli override or cfg.sft_checkpoint)."""
    raw = sft_checkpoint if sft_checkpoint is not None else cfg.get("sft_checkpoint")
    if not raw:
        raise ValueError(
            "sft_checkpoint is required to init dpo; "
            "pass --sft-checkpoint or set sft_checkpoint in configs/dpo.yaml"
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"sft checkpoint not found: {path}")
    return path


def load_dpo_dataset(processed_path: str | Path):
    """load prepared preference pairs (expects `chosen` / `rejected`)."""
    from datasets import load_from_disk

    path = resolve_path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"dpo processed dataset not found: {path}; "
            "run: python scripts/prepare/dpo.py"
        )
    ds = load_from_disk(str(path))
    missing = [c for c in ("chosen", "rejected") if c not in ds.column_names]
    if missing:
        raise ValueError(f"dpo dataset at {path} missing columns: {missing}")
    print(f"loaded dpo dataset: {path} rows={len(ds)}")
    return ds


def build_dpo_config(
    cfg: dict[str, Any],
    *,
    beta: float,
    run_name: str,
    output_dir: str | Path,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> Any:
    """build trl DPOConfig for one beta arm."""
    from trl import DPOConfig

    out = Path(output_dir)
    if push_to_hub:
        hub_kwargs = hub_trainer_kwargs(
            run_name,
            username=hub_username,
            push_to_hub=True,
        )
    else:
        hub_kwargs = {"push_to_hub": False}

    return DPOConfig(
        output_dir=str(out),
        run_name=run_name,
        beta=float(beta),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
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
        max_prompt_length=int(cfg.get("max_prompt_length", 1024)),
        precompute_ref_log_probs=bool(cfg.get("precompute_ref_log_probs", True)),
        remove_unused_columns=False,
        average_tokens_across_devices=False,
        **hub_kwargs,
    )


def build_dpo_trainer(
    cfg: dict[str, Any],
    *,
    beta: float | None = None,
    sft_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
) -> tuple[Any, str, Path, float]:
    """construct DPOTrainer; returns (trainer, run_name, output_dir, beta)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    beta_value = resolve_beta(cfg, beta)
    sft_path = resolve_sft_checkpoint(cfg, sft_checkpoint)
    base_name = cfg["base_name"]
    task = format_dpo_task(beta_value)
    run = run_name or make_run_name(base_name, task)
    out = Path(output_dir) if output_dir is not None else checkpoint_dir(run)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    train_ds = dataset if dataset is not None else load_dpo_dataset(cfg["processed_path"])

    print(f"loading tokenizer/model from sft checkpoint: {sft_path} beta={beta_value:g}")
    tokenizer = AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(sft_path),
        trust_remote_code=True,
        torch_dtype="auto",
    )
    # separate ref copy; with precompute_ref_log_probs trl scores once then can free
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(sft_path),
        trust_remote_code=True,
        torch_dtype="auto",
    )

    args = build_dpo_config(
        cfg,
        beta=beta_value,
        run_name=run,
        output_dir=out,
        hub_username=hub_username,
        push_to_hub=push_to_hub,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    return trainer, run, out, beta_value


def run_dpo(
    cfg: dict[str, Any],
    *,
    beta: float | None = None,
    sft_checkpoint: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    dataset=None,
    hub_username: str | None = None,
    push_to_hub: bool = True,
    wandb_project: str | None = None,
) -> Path:
    """train one dpo beta arm: auto-resume, wandb, train, save, optional hub push."""
    import wandb

    trainer, run, out, beta_value = build_dpo_trainer(
        cfg,
        beta=beta,
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
        wandb.config.update({"beta": beta_value}, allow_val_change=True)

    resume_ckpt = resolve_resume_from_checkpoint(out, resume=True)
    print(
        f"starting dpo run_name={run} beta={beta_value:g} "
        f"output_dir={out} resume={resume_ckpt}"
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out))
    trainer.processing_class.save_pretrained(str(out))

    if push_to_hub:
        push_checkpoint_to_hub(out, run_name=run, username=hub_username)

    if use_wandb:
        wandb.finish()
    print(f"dpo done: {out}")
    return out

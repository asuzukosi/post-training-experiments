"""ppo checkpoint resolution and the four models a ppo step needs.

ppo carries policy, frozen reference, reward and value models at once — the reason it
costs several times an sft step in both memory and wall-clock.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.paths import resolve_path


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


def load_ppo_models(
    *,
    sft_path: Path,
    rm_path: Path,
    policy_path: Path,
    pad_token_id: int,
) -> tuple[Any, Any, Any, Any]:
    """load (policy, ref_policy, reward_model, value_model).

    policy_path differs from sft_path only when resuming from a step checkpoint; the
    reference always stays the sft model, since kl is measured against the sft policy.
    the value model is a fresh scalar head on the sft backbone.
    """
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    print(f"loading policy from: {policy_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        str(policy_path), trust_remote_code=True, torch_dtype="auto"
    )
    print(f"loading ref policy from sft: {sft_path}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        str(sft_path), trust_remote_code=True, torch_dtype="auto"
    )
    print(f"loading reward model from: {rm_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(rm_path), num_labels=1, trust_remote_code=True, torch_dtype="auto"
    )
    reward_model.requires_grad_(False)
    reward_model.config.pad_token_id = pad_token_id

    print(f"loading value model from sft: {sft_path}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        str(sft_path), num_labels=1, trust_remote_code=True, torch_dtype="auto"
    )
    value_model.config.pad_token_id = pad_token_id

    return policy, ref_policy, reward_model, value_model

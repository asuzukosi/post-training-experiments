"""shared huggingface hub helpers for private checkpoint pushes.

spec: push_to_hub=True, hub_model_id="<user>/<base>_<task>_<datetime>",
hub_private_repo=True; token from $HF_HOME/token (volume-persisted).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.data_tools.naming import make_run_name

DEFAULT_HUB_PRIVATE = True


def hf_home() -> Path:
    """return HF_HOME (defaults to ~/.cache/huggingface if unset)."""
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "huggingface"


def read_hf_token(*, token_path: str | Path | None = None) -> str:
    """read write-scope hf token from env or `$HF_HOME/token`."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    path = Path(token_path) if token_path is not None else hf_home() / "token"
    if not path.is_file():
        raise FileNotFoundError(
            f"hf token not found at {path}; run fine-tuning/setup_secrets.sh "
            "or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN"
        )
    token = path.read_text().strip()
    if not token:
        raise ValueError(f"hf token file is empty: {path}")
    return token


def resolve_hub_username(
    username: str | None = None,
    *,
    token: str | None = None,
) -> str:
    """return explicit username, else HF_HUB_USER / HF_USERNAME, else whoami(token)."""
    if username and username.strip():
        return username.strip()
    for key in ("HF_HUB_USER", "HF_USERNAME"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    from huggingface_hub import whoami

    info = whoami(token=token or read_hf_token())
    name = info.get("name") if isinstance(info, dict) else None
    if not name:
        raise ValueError("could not resolve hub username from whoami(); pass username=")
    return str(name)


def make_hub_model_id(username: str, run_name: str) -> str:
    """build `<username>/<run_name>` (run_name is already base_task_stamp)."""
    user = (username or "").strip().strip("/")
    name = (run_name or "").strip().strip("/")
    if not user:
        raise ValueError("username must be non-empty")
    if not name:
        raise ValueError("run_name must be non-empty")
    if "/" in name:
        raise ValueError(f"run_name must not contain '/': {run_name!r}")
    return f"{user}/{name}"


def hub_model_id_for_run(
    base_name: str,
    task: str,
    *,
    username: str | None = None,
    when: datetime | None = None,
    token: str | None = None,
) -> str:
    """build `<user>/<base>_<task>_<datetime>` from naming helpers."""
    user = resolve_hub_username(username, token=token)
    return make_hub_model_id(user, make_run_name(base_name, task, when=when))


def hub_trainer_kwargs(
    run_name: str,
    *,
    username: str | None = None,
    private: bool = DEFAULT_HUB_PRIVATE,
    push_to_hub: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """kwargs to merge into transformers/trl TrainingArguments for hub push."""
    tok = token if token is not None else read_hf_token()
    user = resolve_hub_username(username, token=tok)
    return {
        "push_to_hub": push_to_hub,
        "hub_model_id": make_hub_model_id(user, run_name),
        "hub_private_repo": private,
        "hub_token": tok,
    }


def push_checkpoint_to_hub(
    local_dir: str | Path,
    *,
    run_name: str | None = None,
    repo_id: str | None = None,
    username: str | None = None,
    private: bool = DEFAULT_HUB_PRIVATE,
    token: str | None = None,
    commit_message: str = "upload checkpoint",
) -> str:
    """upload a local checkpoint folder to a private hub repo; return repo_id.

    pass either `repo_id` or `run_name` (with optional username).
    """
    from huggingface_hub import HfApi

    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {local}")

    tok = token if token is not None else read_hf_token()
    if repo_id:
        target = repo_id.strip().strip("/")
    else:
        if not run_name:
            raise ValueError("pass repo_id= or run_name=")
        target = make_hub_model_id(resolve_hub_username(username, token=tok), run_name)

    api = HfApi(token=tok)
    api.create_repo(repo_id=target, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(local),
        repo_id=target,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"pushed {local} -> {target} (private={private})")
    return target

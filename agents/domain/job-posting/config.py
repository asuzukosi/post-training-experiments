"""local env/model helpers for this project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR
for parent in [_APP_DIR, *_APP_DIR.parents]:
    if (parent / "pyproject.toml").exists() or (parent / ".env.example").exists():
        _ROOT_DIR = parent
        break

DEFAULT_GPT_MODEL = "gpt-5.6-luna"

load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_APP_DIR / ".env", override=True)


def get_model_name() -> str:
    name = (os.getenv("MODEL_NAME") or DEFAULT_GPT_MODEL).strip().strip("'\"")
    if ":" in name:
        _, name = name.split(":", 1)
    lowered = name.lower()
    if "hermes" in lowered or "ollama" in lowered or "llama" in lowered:
        return DEFAULT_GPT_MODEL
    if not lowered.startswith("gpt"):
        return DEFAULT_GPT_MODEL
    return name


def get_model_id(model_name: str | None = None) -> str:
    raw = model_name if model_name is not None else (os.getenv("MODEL_NAME") or DEFAULT_GPT_MODEL)
    name = raw.strip().strip("'\"")
    if ":" in name:
        _, name = name.split(":", 1)
    lowered = name.lower()
    if "hermes" in lowered or "ollama" in lowered or "llama" in lowered or not lowered.startswith("gpt"):
        name = DEFAULT_GPT_MODEL
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or name
        )
        return f"azure_openai:{deployment}"
    return f"openai:{name}"


def final_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)

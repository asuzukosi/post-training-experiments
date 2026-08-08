"""shared config for the multi-doc agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = APP_DIR / "data"


def get_model_name() -> str:
    return os.getenv("MODEL_NAME") or "gpt-5.6-luna"


def _guess_provider(name: str) -> str:
    if "gpt" in name:
        return "openai"
    if "gemini" in name:
        return "google"
    if "claude" in name:
        return "anthropic"
    if "llama" in name:
        return "meta"
    return "openai"


def get_model_id() -> str:
    """provider:model id for langchain / deepagents."""
    name = get_model_name()
    if ":" in name:
        return name
    return f"{_guess_provider(name)}:{name}"


DEFAULT_MODEL = get_model_name()

"""shared config for the data science agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent.parent

# shared secrets first, then project overrides
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_APP_DIR / ".env", override=True)


def get_model_name() -> str:
    return os.getenv("MODEL_NAME") or "gpt-5.6-luna"


def _guess_provider(name: str) -> str:
    """guess the provider from the model name."""
    if "gpt" in name:
        return "openai"
    if "gemini" in name:
        return "google"
    if "claude" in name:
        return "anthropic"
    if "llama" in name:
        return "meta"
    return "openai"


def get_model_id(model_name: str | None = None) -> str:
    """provider:model id for langchain / deepagents."""
    name = model_name or get_model_name()
    if ":" in name:
        return name
    return f"{_guess_provider(name)}:{name}"


DEFAULT_MODEL = get_model_name()

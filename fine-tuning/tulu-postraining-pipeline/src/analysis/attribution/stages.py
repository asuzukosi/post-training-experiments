"""which stages the pipeline produces, and which are preference stages."""
from __future__ import annotations

GENERATIVE_STAGES = (
    "base",
    "sft",
    "dpo-b0.05",
    "dpo-b0.1",
    "ppo",
)
PREFERENCE_STAGES = ("dpo-b0.05", "dpo-b0.1", "ppo")

"""markdown validator agent via create_agent."""
from __future__ import annotations
import sys
from textwrap import dedent
from langchain.agents import create_agent

from config import final_text, get_model_id
from tools import markdown_validation_tool


SYSTEM_PROMPT = dedent(
    """
    you are a requirements manager and expert business analyst / software qa specialist.
    provide a detailed list of markdown linting results and a summary with actionable
    tasks for a developer to fix the issues.
    do not provide examples of how to fix issues or recommend other tools.
    do not change document content — only list required changes.
    """
).strip()


def process_markdown_document(filename: str) -> str:
    agent = create_agent(
        model=get_model_id(),
        tools=[markdown_validation_tool],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"use the markdown_validation_tool to review the file at: {filename}. "
                        "pass only the file path to the tool. then summarize validation results "
                        "into a list of changes the developer should make."
                    ),
                }
            ]
        }
    )
    return final_text(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(process_markdown_document(sys.argv[1]))
    else:
        print("usage: python main.py <markdown-file>")

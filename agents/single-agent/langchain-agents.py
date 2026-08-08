"""single-agent demo using langchain create_agent."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool

from config import final_text, get_model_id



@tool
def search(query: str) -> str:
    """search things about current events."""
    return "32 degrees"


def main() -> None:
    agent = create_agent(
        model=get_model_id(),
        tools=[search],
        system_prompt="you are a helpful assistant. use tools when needed.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "whats the weather in new york?"}]}
    )
    print(final_text(result))


if __name__ == "__main__":
    main()

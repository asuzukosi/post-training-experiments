"""azure / openai researcher demo via create_agent."""

from __future__ import annotations
from textwrap import dedent

from langchain.agents import create_agent

from config import final_text, get_model_id


SYSTEM_PROMPT = dedent(
    """
    you are a senior researcher. a curious mind fascinated by cutting-edge
    innovation and the potential to change the world. you know everything about tech.
    discover groundbreaking technologies. be concise.
    """
).strip()


def main() -> None:
    agent = create_agent(
        model=get_model_id(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "identify the next big trend in ai"}]}
    )
    print(final_text(result))


if __name__ == "__main__":
    main()

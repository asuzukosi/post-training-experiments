"""stock chart coding agent via create_deep_agent (replaces autogen quickstart)."""

from __future__ import annotations
from deepagents import create_deep_agent
from config import final_text, get_model_id


SYSTEM_PROMPT = (
    "you are a coding assistant. write and explain python to solve the user's request. "
    "prefer clear, runnable code."
)


def main() -> None:
    agent = create_deep_agent(
        model=get_model_id(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "plot a chart of nvda and tesla stock price change ytd. "
                    "show the python code to do it with yfinance and matplotlib.",
                }
            ]
        }
    )
    print(final_text(result))


if __name__ == "__main__":
    main()

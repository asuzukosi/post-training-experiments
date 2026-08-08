"""coding deep agent."""

from __future__ import annotations
import sys
from deepagents import create_deep_agent
from config import final_text, get_model_id



def main() -> None:
    agent = create_deep_agent(
        model=get_model_id(),
        tools=[],
        system_prompt=(
            "you are an expert software engineer. solve coding tasks with clear "
            "python, explain briefly, and prefer correct runnable solutions."
        ),
    )
    query = (
        "write a python script that downloads daily close prices for a ticker "
        "with yfinance and prints the last 5 rows."
    )
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(final_text(result))


if __name__ == "__main__":
    main()

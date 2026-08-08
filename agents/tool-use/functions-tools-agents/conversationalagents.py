"""conversational tool-calling agent via create_agent."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from config import final_text, get_model_id



class WeatherInput(BaseModel):
    latitude: float = Field(description="latitude")
    longitude: float = Field(description="longitude")


@tool(args_schema=WeatherInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """fetch current temperature for coordinates (stub)."""
    return f"stub temperature at ({latitude}, {longitude}): 18°c"


@tool
def search_wikipedia(query: str) -> str:
    """search wikipedia (stub)."""
    return f"stub wikipedia summary for: {query}"


def main() -> None:
    agent = create_agent(
        model=get_model_id(),
        tools=[get_current_temperature, search_wikipedia],
        system_prompt="helpful assistant. use tools when needed.",
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what is the temperature in san francisco (37.77, -122.42)?",
                }
            ]
        }
    )
    print(final_text(result))


if __name__ == "__main__":
    main()

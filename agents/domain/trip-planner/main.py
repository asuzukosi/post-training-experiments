"""trip planner: sequential langgraph pipeline."""

from __future__ import annotations

from textwrap import dedent
from pipeline import build_sequential_graph, make_agent_node
from tools.browser_tools import scrape_and_summarize_website
from tools.calculator_tools import calculate
from tools.search_tools import search_internet


TIP = "if you do your best work, i'll tip you $100!"

CITY_SELECTOR_PROMPT = dedent(
    """
    you are a city selection expert.
    goal: select the best city based on weather, season, and prices.
    backstory: an expert in analyzing travel data to pick ideal destinations.
    """
).strip()

LOCAL_EXPERT_PROMPT = dedent(
    """
    you are a local expert at the selected city.
    goal: provide the best insights about the selected city.
    backstory: a knowledgeable local guide with extensive information about the city,
    its attractions, and customs.
    """
).strip()

TRAVEL_CONCIERGE_PROMPT = dedent(
    """
    you are an amazing travel concierge.
    goal: create the most amazing travel itineraries with budget and packing suggestions.
    backstory: specialist in travel planning and logistics with decades of experience.
    """
).strip()

RESEARCH_TOOLS = [search_internet, scrape_and_summarize_website]
PLANNING_TOOLS = [search_internet, scrape_and_summarize_website, calculate]


def build_graph(origin: str, cities: str, date_range: str, interests: str):
    identify_prompt = (
        f"{CITY_SELECTOR_PROMPT}\n\n"
        "analyze and select the best city for the trip based on weather patterns, "
        "seasonal events, and travel costs. compare multiple cities and include "
        "flight costs, weather forecast, and attractions.\n\n"
        f"traveling from: {origin}\n"
        f"city options: {cities}\n"
        f"trip date: {date_range}\n"
        f"traveler interests: {interests}\n\n"
        f"{TIP}"
    )
    gather_prompt = (
        f"{LOCAL_EXPERT_PROMPT}\n\n"
        "compile an in-depth city guide with key attractions, local customs, "
        "special events, hidden gems, weather forecasts, and high level costs.\n\n"
        f"trip date: {date_range}\n"
        f"traveling from: {origin}\n"
        f"traveler interests: {interests}\n\n"
        f"{TIP}"
    )
    plan_prompt = (
        f"{TRAVEL_CONCIERGE_PROMPT}\n\n"
        "expand the city guide into a full 7-day travel itinerary with per-day plans, "
        "weather forecasts, places to eat, packing suggestions, and a budget breakdown. "
        "suggest actual places, hotels, and restaurants. output as markdown.\n\n"
        f"trip date: {date_range}\n"
        f"traveling from: {origin}\n"
        f"traveler interests: {interests}\n\n"
        f"{TIP}"
    )
    return build_sequential_graph(
        [
            ("city_selection", make_agent_node(identify_prompt, tools=RESEARCH_TOOLS)),
            ("local_guide", make_agent_node(gather_prompt, tools=RESEARCH_TOOLS)),
            ("itinerary", make_agent_node(plan_prompt, tools=PLANNING_TOOLS)),
        ]
    )


def main() -> None:
    print("## welcome to trip planner")
    print("-------------------------------")
    origin = input("from where will you be traveling from?\n").strip()
    cities = input("what cities are you interested in visiting?\n").strip()
    date_range = input("what is the date range you are interested in traveling?\n").strip()
    interests = input("what are some of your high level interests and hobbies?\n").strip()

    graph = build_graph(origin, cities, date_range, interests)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"plan a trip from {origin} to one of: {cities}. "
                        f"dates: {date_range}. interests: {interests}."
                    ),
                }
            ],
            "output": "",
        }
    )

    print("\n\n########################")
    print("## here is your trip plan")
    print("########################\n")
    print(result.get("output", result))


if __name__ == "__main__":
    main()

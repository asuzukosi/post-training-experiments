"""instagram post marketing: sequential langgraph pipeline (copy + image phases)."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node
from tools.browser_tools import scrape_and_summarize_website
from tools.search_tools import search_internet, search_instagram


MARKET_ANALYST_PROMPT = dedent(
    """
    you are the lead market analyst at a premier digital marketing firm.
    goal: conduct amazing analysis of products and competitors with in-depth insights.
    backstory: you specialize in dissecting online business landscapes.
    """
).strip()

STRATEGY_PROMPT = dedent(
    """
    you are the chief marketing strategist at a leading digital marketing agency.
    goal: synthesize product analysis into incredible marketing strategies.
    backstory: you craft bespoke strategies that drive success.
    """
).strip()

CREATIVE_PROMPT = dedent(
    """
    you are a creative content creator at a top-tier digital marketing agency.
    goal: develop compelling content for social media, especially high-impact instagram ad copy.
    backstory: you turn marketing strategies into engaging stories that capture attention.
    """
).strip()

PHOTOGRAPHER_PROMPT = dedent(
    """
    you are a senior photographer at a leading digital marketing agency.
    goal: describe amazing photographs for instagram ads that capture emotions and convey a message.
    backstory: you inspire and engage through visual storytelling for important campaigns.
    """
).strip()

CREATIVE_DIRECTOR_PROMPT = dedent(
    """
    you are the chief creative director specializing in product branding.
    goal: review and refine photograph concepts so they align with product goals.
    backstory: you ensure your team delivers the best possible content for each customer.
    """
).strip()

PHOTO_EXAMPLES = dedent(
    """
    - high tech airplane in a beautiful blue sky at sunset, super crisp 4k, professional wide shot
    - the last supper with jesus and disciples breaking bread, close shot, soft lighting, 4k, crisp
    - a bearded old man in the snow using warm clothing, mountains behind, soft lighting, 4k, close up
    """
).strip()

RESEARCH_TOOLS = [scrape_and_summarize_website, search_internet]
STRATEGY_TOOLS = [scrape_and_summarize_website, search_internet, search_instagram]
CREATIVE_TOOLS = STRATEGY_TOOLS


def build_graph(product_website: str, product_details: str):
    product_analysis_prompt = (
        f"{MARKET_ANALYST_PROMPT}\n\n"
        f"analyze the product website: {product_website}.\n"
        f"extra details: {product_details}.\n\n"
        "identify unique features, benefits, and narrative. articulate key selling points, "
        "market appeal, and positioning suggestions. it is currently 2026."
    )
    competitor_prompt = (
        f"{MARKET_ANALYST_PROMPT}\n\n"
        f"explore competitors of: {product_website}.\n"
        f"extra details: {product_details}.\n\n"
        "identify the top 3 competitors and analyze strategies, positioning, and perception. "
        f"include context about {product_website} and a detailed competitor comparison."
    )
    campaign_prompt = (
        f"{STRATEGY_PROMPT}\n\n"
        f"create a targeted marketing campaign for: {product_website}.\n"
        f"extra details: {product_details}.\n\n"
        "design a strategy and creative content ideas that captivate the target audience. "
        "include all context about the product and customer."
    )
    ad_copy_prompt = (
        f"{CREATIVE_PROMPT}\n\n"
        "craft engaging instagram post copy aligned with the marketing strategy from prior steps. "
        "output 3 ad copy options that inform, excite, and persuade."
    )
    photograph_prompt = (
        f"{PHOTOGRAPHER_PROMPT}\n\n"
        f"product: {product_website}. extra details: {product_details}.\n\n"
        "using the ad copy from the previous step, describe 3 photograph options for the post. "
        "each option is one paragraph. do not show the actual product in the photo.\n\n"
        f"examples:\n{PHOTO_EXAMPLES}"
    )
    review_prompt = (
        f"{CREATIVE_DIRECTOR_PROMPT}\n\n"
        f"product: {product_website}. extra details: {product_details}.\n\n"
        "review the photographer's concepts from the previous step. output 3 reviewed photograph "
        "options, each with one paragraph following these examples:\n\n"
        f"{PHOTO_EXAMPLES}"
    )
    return build_sequential_graph(
        [
            ("product_analysis", make_agent_node(product_analysis_prompt, tools=RESEARCH_TOOLS)),
            ("competitor_analysis", make_agent_node(competitor_prompt, tools=RESEARCH_TOOLS)),
            ("campaign_development", make_agent_node(campaign_prompt, tools=STRATEGY_TOOLS)),
            ("instagram_ad_copy", make_agent_node(ad_copy_prompt, tools=CREATIVE_TOOLS)),
            ("photograph_concepts", make_agent_node(photograph_prompt, tools=CREATIVE_TOOLS)),
            ("review_photographs", make_agent_node(review_prompt, tools=CREATIVE_TOOLS)),
        ]
    )


def main() -> None:
    print("## welcome to the instagram marketing pipeline")
    print("-------------------------------")
    product_website = input(
        "what is the product website you want a marketing strategy for?\n"
    ).strip()
    product_details = input(
        "any extra details about the product and or the instagram post you want?\n"
    ).strip()

    graph = build_graph(product_website, product_details)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"create an instagram marketing plan for {product_website}. "
                        f"details: {product_details}."
                    ),
                }
            ],
            "output": "",
        }
    )

    output = result.get("output", result)
    print("\n\n########################")
    print("## here is the result")
    print("########################\n")
    print("final output (ad copy + midjourney-style photo descriptions):")
    print(output)


if __name__ == "__main__":
    main()

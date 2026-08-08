"""stock analysis: sequential langgraph pipeline with financial tools."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools


TIP = "if you do your best work, i will give you a $10,000 commission!"

RESEARCH_TOOLS = [
    BrowserTools.scrape_and_summarize_website,
    SearchTools.search_internet,
    SearchTools.search_news,
    SECTools.search_10q,
    SECTools.search_10k,
]

FINANCIAL_TOOLS = [
    BrowserTools.scrape_and_summarize_website,
    SearchTools.search_internet,
    CalculatorTools.calculate,
    SECTools.search_10q,
    SECTools.search_10k,
]

ADVISOR_TOOLS = [
    BrowserTools.scrape_and_summarize_website,
    SearchTools.search_internet,
    SearchTools.search_news,
    CalculatorTools.calculate,
]

RESEARCH_PROMPT = dedent(
    """
    you are the best staff research analyst, skilled in news, announcements, and market sentiment.
    collect and summarize recent news, press releases, and market analyses for the company.
    pay attention to significant events, sentiment, analyst opinions, and upcoming earnings.
    your final answer must be a comprehensive summary including the stock ticker.
    use the most recent data possible. {tip}
    """
).strip()

FINANCIAL_PROMPT = dedent(
    """
    you are the best financial analyst, expert in stock metrics and market trends.
    conduct a thorough analysis of financial health and market performance.
    examine p/e ratio, eps growth, revenue trends, debt-to-equity, and peer comparison.
    expand on prior research with a clear assessment of strengths, weaknesses, and competitors.
    use the most recent data possible. {tip}
    """
).strip()

FILINGS_PROMPT = dedent(
    """
    you are the best financial analyst analyzing sec edgar filings.
    analyze the latest 10-q and 10-k filings. focus on md&a, financial statements,
    insider trading, and disclosed risks. highlight red flags and positive indicators.
    {tip}
    """
).strip()

RECOMMEND_PROMPT = dedent(
    """
    you are the most experienced private investment advisor.
    synthesize all prior analyses into a comprehensive investment recommendation.
    include insider trading activity and upcoming events like earnings.
    provide a clear investment stance and strategy with supporting evidence.
    format the report well for the customer. {tip}
    """
).strip()


def build_stock_graph(company: str):
    research_prompt = (
        f"{RESEARCH_PROMPT.format(tip=TIP)}\n\nselected company: {company}"
    )
    return build_sequential_graph(
        [
            ("research", make_agent_node(research_prompt, tools=RESEARCH_TOOLS)),
            ("financial", make_agent_node(FINANCIAL_PROMPT.format(tip=TIP), tools=FINANCIAL_TOOLS)),
            ("filings", make_agent_node(FILINGS_PROMPT.format(tip=TIP), tools=FINANCIAL_TOOLS)),
            ("recommend", make_agent_node(RECOMMEND_PROMPT.format(tip=TIP), tools=ADVISOR_TOOLS)),
        ]
    )

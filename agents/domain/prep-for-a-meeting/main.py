"""prep for a meeting: sequential langgraph pipeline."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node
from tools.exa_search_tools import exa_tools


RESEARCH_PROMPT = dedent(
    """
    you are a research specialist.
    goal: conduct thorough research on people and companies involved in the meeting.
    backstory: your mission is to uncover detailed information about the individuals
    and entities participating in the meeting. your insights lay the groundwork
    for strategic meeting preparation.
    """
).strip()

INDUSTRY_ANALYSIS_PROMPT = dedent(
    """
    you are an industry analyst.
    goal: analyze current industry trends, challenges, and opportunities.
    backstory: your analysis identifies key trends, challenges facing the industry,
    and potential opportunities that could be leveraged during the meeting.
    """
).strip()

MEETING_STRATEGY_PROMPT = dedent(
    """
    you are a meeting strategy advisor.
    goal: develop talking points, questions, and strategic angles for the meeting.
    backstory: your expertise guides the development of talking points, insightful
    questions, and strategic angles to ensure the meeting objectives are achieved.
    """
).strip()

BRIEFING_PROMPT = dedent(
    """
    you are a briefing coordinator.
    goal: compile all gathered information into a concise, informative briefing document.
    backstory: you consolidate research, analysis, and strategic insights into one briefing.
    """
).strip()


def build_graph(participants: str, context: str, objective: str):
    research_prompt = (
        f"{RESEARCH_PROMPT}\n\n"
        "conduct comprehensive research on each individual and company involved "
        "in the upcoming meeting. gather information on recent news, achievements, "
        "professional background, and relevant business activities.\n\n"
        f"participants: {participants}\n"
        f"meeting context: {context}\n\n"
        "expected output: a detailed report summarizing key findings about each "
        "participant and company, highlighting information relevant for the meeting."
    )
    industry_prompt = (
        f"{INDUSTRY_ANALYSIS_PROMPT}\n\n"
        "analyze current industry trends, challenges, and opportunities relevant "
        "to the meeting context. consider market reports, recent developments, "
        "and expert opinions.\n\n"
        f"participants: {participants}\n"
        f"meeting context: {context}\n\n"
        "expected output: an insightful analysis identifying major trends, "
        "potential challenges, and strategic opportunities."
    )
    strategy_prompt = (
        f"{MEETING_STRATEGY_PROMPT}\n\n"
        "develop strategic talking points, questions, and discussion angles "
        "based on prior research and industry analysis.\n\n"
        f"meeting context: {context}\n"
        f"meeting objective: {objective}\n\n"
        "expected output: key talking points and strategic questions to achieve "
        "the meeting objective."
    )
    briefing_prompt = (
        f"{BRIEFING_PROMPT}\n\n"
        "compile all research, industry analysis, and strategic talking points "
        "into a concise briefing document.\n\n"
        f"meeting context: {context}\n"
        f"meeting objective: {objective}\n\n"
        "expected output: a well-structured briefing with participant bios, "
        "industry overview, talking points, and strategic recommendations."
    )
    return build_sequential_graph(
        [
            ("research", make_agent_node(research_prompt, tools=exa_tools)),
            ("industry_analysis", make_agent_node(industry_prompt, tools=exa_tools)),
            ("meeting_strategy", make_agent_node(strategy_prompt, tools=exa_tools)),
            ("briefing", make_agent_node(briefing_prompt, tools=exa_tools)),
        ]
    )


def main() -> None:
    print("## welcome to the meeting prep pipeline")
    print("-------------------------------")
    participants = input(
        "what are the emails for the participants (other than you) in the meeting?\n"
    ).strip()
    context = input("what is the context of the meeting?\n").strip()
    objective = input("what is your objective for this meeting?\n").strip()

    graph = build_graph(participants, context, objective)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"prepare me for a meeting. participants: {participants}. "
                        f"context: {context}. objective: {objective}."
                    ),
                }
            ],
            "output": "",
        }
    )

    print("\n\n################################################")
    print("## here is the result")
    print("################################################\n")
    print(result.get("output", result))


if __name__ == "__main__":
    main()

"""sequential email drafting pipeline (replaces crewai crew node)."""

from __future__ import annotations

from textwrap import dedent

from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools.tavily_search import TavilySearchResults

from pipeline import build_sequential_graph, make_agent_node

from .tools import CreateDraftTool


FILTER_PROMPT = dedent(
    """
    you are a senior email analyst.
    analyze a batch of emails and filter out non-essential ones such as newsletters,
    promotional content and notifications.
    distinguish important emails from spam. pay attention to the sender.
    filter for messages actually directed at the user and avoid notifications.
    your final answer must be the relevant thread_ids and the sender, use bullet points.
    """
).strip()

ACTION_PROMPT = dedent(
    """
    you are an email action specialist.
    for each email thread, pull and analyze the complete threads using the thread id tool.
    understand context, key points, and overall sentiment.
    identify the main query or concerns to address in the response for each.
    your final answer must be a list for all emails with thread_id, summary, main points,
    who the user is answering, communication style, and sender email address.
    """
).strip()

DRAFT_PROMPT = dedent(
    """
    you are an email response writer.
    based on the action-required emails identified, draft responses for each.
    assume the persona of the user and mimic the communication style in the thread.
    research the topic if necessary before drafting.
    use the create draft tool for each response with to, subject, and message.
    you must create all drafts before your final answer.
    your final answer must confirm that all responses have been drafted.
    """
).strip()


def _format_emails(emails: list[dict]) -> str:
    parts = []
    for email in emails:
        print(email)
        parts.append(
            "\n".join(
                [
                    f"id: {email['id']}",
                    f"- thread id: {email['threadId']}",
                    f"- snippet: {email['snippet']}",
                    f"- from: {email['sender']}",
                    "--------",
                ]
            )
        )
    return "\n".join(parts)


def draft_responses_node(state: dict) -> dict:
    print("### filtering emails")
    gmail = GmailToolkit()
    action_tools = [
        GmailGetThread(api_resource=gmail.api_resource),
        TavilySearchResults(),
    ]
    draft_tools = [
        TavilySearchResults(),
        GmailGetThread(api_resource=gmail.api_resource),
        CreateDraftTool.create_draft,
    ]

    emails_str = _format_emails(state["emails"])
    graph = build_sequential_graph(
        [
            ("filter", make_agent_node(FILTER_PROMPT)),
            ("action", make_agent_node(ACTION_PROMPT, tools=action_tools)),
            ("draft", make_agent_node(DRAFT_PROMPT, tools=draft_tools)),
        ]
    )
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": emails_str}],
            "output": emails_str,
        }
    )
    return {**state, "action_required_emails": result.get("output", "")}

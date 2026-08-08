"""deep agent entrypoint built on langchain-ai deepagents (langgraph runtime)."""

from __future__ import annotations

from textwrap import dedent

from deepagents import create_deep_agent
from langchain.messages import AIMessage, ToolMessage

from config import get_model_id
from tools import AGENT_TOOLS

SYSTEM_PROMPT = dedent(
    """
    you are a data science agent for vehicle telemetry and fleet analytics.

    use tools to:
    - answer dataset questions
    - draw charts from the dataset
    - predict vehicle velocity or driving condition
    - search the web when needed

    prefer tools over guessing. be concise.
    """
).strip()


def build_agent(model_id: str | None = None):
    """create a deep agent with domain tools."""
    return create_deep_agent(
        model=model_id or get_model_id(),
        tools=AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def _format_tool_meta(messages) -> str:
    parts = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                parts.append(
                    dedent(
                        f"""
                        function name : {call.get("name")}
                        function arguments : {call.get("args")}
                        """
                    ).strip()
                )
        elif isinstance(message, ToolMessage):
            parts.append(f"tool result : {message.content}")
    return "\n\n".join(parts) if parts else "No function call"


def run_agent_executor(query: str) -> tuple[str, str]:
    """run the deep agent. returns (final_text, tool_metadata)."""
    if isinstance(query, dict):
        query = query.get("content", str(query))

    result = get_agent().invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])
    final = messages[-1].content if messages else ""
    if isinstance(final, list):
        final = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in final
        )
    return str(final), _format_tool_meta(messages)

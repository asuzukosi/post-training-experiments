"""sequential langgraph pipeline helpers for this project."""

from __future__ import annotations

from typing import Annotated, Any, Callable, TypedDict

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import final_text, get_model_id


class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    output: str


def make_agent_node(system_prompt: str, tools: list | None = None, model_id: str | None = None):
    """create a graph node that runs a create_agent and stores final text in output."""
    agent = create_agent(
        model=model_id or get_model_id(),
        tools=tools or [],
        system_prompt=system_prompt,
    )

    def node(state: PipelineState) -> dict[str, Any]:
        prior = state.get("output") or ""
        user_messages = state.get("messages") or []
        if prior:
            invoke_messages = [{"role": "user", "content": prior}]
        else:
            invoke_messages = user_messages
        result = agent.invoke({"messages": invoke_messages})
        text = final_text(result)
        return {"output": text, "messages": [{"role": "assistant", "content": text}]}

    return node


def build_sequential_graph(nodes: list[tuple[str, Callable]]):
    """build a linear StateGraph from (name, node_fn) pairs. node names must be unique."""
    if not nodes:
        raise ValueError("nodes must not be empty")

    names = [name for name, _ in nodes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"node names must be unique; duplicates: {duplicates}")

    graph = StateGraph(PipelineState)
    for name, fn in nodes:
        graph.add_node(name, fn)
    graph.add_edge(START, names[0])
    for left, right in zip(names, names[1:]):
        graph.add_edge(left, right)
    graph.add_edge(names[-1], END)
    return graph.compile()

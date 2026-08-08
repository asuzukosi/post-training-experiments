"""supervisor routes researcher/coder workers via create_agent."""

from __future__ import annotations

import functools
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import final_text, get_model_id


Member = Literal["Researcher", "Coder", "FINISH"]


@tool
def search(query: str) -> str:
    """search the web for up-to-date information."""
    return f"stub results for: {query}"


@tool
def run_python(code: str) -> str:
    """run short python code and return stdout."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, {})
    except Exception as exc:
        return f"error: {exc}"
    return buf.getvalue() or "ok"


@tool
def route(next_worker: Member) -> str:
    """pick the next worker or FINISH when the task is done."""
    return next_worker


def _route_from_result(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        call = tool_calls[0]
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        if isinstance(args, str):
            return args
        return args.get("next_worker") or args.get("next") or "FINISH"
    return "FINISH"


def make_worker(system_prompt: str, tools: list):
    return create_agent(model=get_model_id(), tools=tools, system_prompt=system_prompt)


class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


def worker_node(state: TeamState, agent, name: str) -> dict:
    result = agent.invoke({"messages": state["messages"]})
    text = final_text(result)
    return {"messages": [{"role": "assistant", "content": f"[{name}] {text}"}]}


def supervisor_node(state: TeamState, supervisor) -> dict:
    result = supervisor.invoke({"messages": state["messages"]})
    return {"next": _route_from_result(result)}


def build_graph():
    researcher = make_worker("you are a web researcher. use search when needed.", [search])
    coder = make_worker("you are a coder. use run_python for short scripts.", [run_python])
    supervisor = create_agent(
        model=get_model_id(),
        tools=[route],
        system_prompt=(
            "supervise Researcher (search) and Coder (python). "
            "call route with the next worker, or FINISH when done."
        ),
    )

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", functools.partial(supervisor_node, supervisor=supervisor))
    graph.add_node("Researcher", functools.partial(worker_node, agent=researcher, name="Researcher"))
    graph.add_node("Coder", functools.partial(worker_node, agent=coder, name="Coder"))
    graph.add_edge(START, "supervisor")
    graph.add_edge("Researcher", "supervisor")
    graph.add_edge("Coder", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"Researcher": "Researcher", "Coder": "Coder", "FINISH": END},
    )
    return graph.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "code hello world and print it"}],
            "next": "Researcher",
        },
        config={"recursion_limit": 12},
    )
    for msg in result["messages"]:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "message")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        print(f"{role}: {content}")


if __name__ == "__main__":
    main()

"""hierarchical research + writing teams using create_agent subgraphs."""

from __future__ import annotations

import functools
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import final_text, get_model_id


Route = Literal["Search", "Writer", "FINISH"]


@tool
def search(query: str) -> str:
    """search the web."""
    return f"stub search summary for: {query}"


@tool
def write_outline(topic: str) -> str:
    """draft a short outline."""
    return f"outline for {topic}: intro, findings, conclusion"


@tool
def route(next_worker: Route) -> str:
    """route to a team worker or FINISH."""
    return next_worker


def _route_from_result(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        tool_calls = getattr(msg, "tool_calls", None) or (
            msg.get("tool_calls") if isinstance(msg, dict) else None
        )
        if not tool_calls:
            continue
        call = tool_calls[0]
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        return args.get("next_worker") or args.get("next") or "FINISH"
    return "FINISH"


class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


def worker_node(state: TeamState, agent, name: str) -> dict:
    result = agent.invoke({"messages": state["messages"]})
    return {"messages": [{"role": "assistant", "content": f"[{name}] {final_text(result)}"}]}


def supervisor_node(state: TeamState, supervisor) -> dict:
    result = supervisor.invoke({"messages": state["messages"]})
    return {"next": _route_from_result(result)}


def build_team_graph(members: list[str], workers: dict, supervisor_prompt: str):
    supervisor = create_agent(
        model=get_model_id(),
        tools=[route],
        system_prompt=supervisor_prompt,
    )
    graph = StateGraph(TeamState)
    graph.add_node("supervisor", functools.partial(supervisor_node, supervisor=supervisor))
    for name, agent in workers.items():
        graph.add_node(name, functools.partial(worker_node, agent=agent, name=name))
        graph.add_edge(name, "supervisor")
    graph.add_edge(START, "supervisor")
    routes = {member: member for member in members}
    routes["FINISH"] = END
    graph.add_conditional_edges("supervisor", lambda state: state["next"], routes)
    return graph.compile()


def main() -> None:
    research_team = build_team_graph(
        members=["Search", "FINISH"],
        workers={
            "Search": create_agent(
                model=get_model_id(),
                tools=[search],
                system_prompt="research assistant: use search for current facts.",
            ),
        },
        supervisor_prompt=(
            "manage Search. route to Search for research, FINISH when enough info exists."
        ),
    )
    writing_team = build_team_graph(
        members=["Writer", "FINISH"],
        workers={
            "Writer": create_agent(
                model=get_model_id(),
                tools=[write_outline],
                system_prompt="writing assistant: produce concise outlines.",
            ),
        },
        supervisor_prompt="manage Writer. route to Writer to draft, FINISH when outline is ready.",
    )

    research = research_team.invoke(
        {
            "messages": [{"role": "user", "content": "when is the next solar eclipse?"}],
            "next": "Search",
        },
        config={"recursion_limit": 8},
    )
    print("research:", final_text({"messages": research["messages"]}))

    writing = writing_team.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"write an outline using: {final_text({'messages': research['messages']})}",
                }
            ],
            "next": "Writer",
        },
        config={"recursion_limit": 8},
    )
    print("writing:", final_text({"messages": writing["messages"]}))


if __name__ == "__main__":
    main()

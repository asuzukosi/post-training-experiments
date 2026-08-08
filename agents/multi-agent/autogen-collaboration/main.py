"""multi-role task solving via langgraph (replaces autogen group chat)."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node


CODER = dedent(
    """
    you are a coder. write clear python solutions for the user's task.
    focus on correct, runnable code.
    """
).strip()

PM = dedent(
    """
    you are a product manager. review the coder's work, clarify requirements,
    and suggest improvements for usability and completeness.
    """
).strip()

MANAGER = dedent(
    """
    you are the project manager. synthesize the coder and pm outputs into a
    final answer for the user. be concise and actionable.
    """
).strip()


def main() -> None:
    graph = build_sequential_graph(
        [
            ("coder", make_agent_node(CODER)),
            ("pm", make_agent_node(PM)),
            ("manager", make_agent_node(MANAGER)),
        ]
    )
    task = (
        "write a small python function that finds the longest common prefix "
        "among a list of strings, with a short explanation."
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": task}], "output": ""}
    )
    print(result.get("output", result))


if __name__ == "__main__":
    main()

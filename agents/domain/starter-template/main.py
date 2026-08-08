"""starter template: sequential two-role langgraph pipeline."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node


AGENT_1_PROMPT = dedent(
    """
    you are agent 1. define agent 1 role here.
    backstory: define agent 1 backstory here.
    goal: define agent 1 goal here.
    do your best work.
    """
).strip()

AGENT_2_PROMPT = dedent(
    """
    you are agent 2. define agent 2 role here.
    backstory: define agent 2 backstory here.
    goal: define agent 2 goal here.
    take the previous agent's output and improve or continue the work.
    do your best work.
    """
).strip()


def build_graph(var1: str, var2: str):
    task1_prompt = (
        f"{AGENT_1_PROMPT}\n\n"
        f"do something as part of task 1.\n"
        f"use this variable: {var1}\n"
        f"and also this variable: {var2}\n"
        "use the most recent data as possible."
    )
    return build_sequential_graph(
        [
            ("agent_1", make_agent_node(task1_prompt)),
            ("agent_2", make_agent_node(AGENT_2_PROMPT)),
        ]
    )


def main() -> None:
    print("## welcome to langgraph agent template")
    print("-------------------------------")
    var1 = input("enter variable 1: ").strip()
    var2 = input("enter variable 2: ").strip()
    graph = build_graph(var1, var2)
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": f"start with var1={var1}, var2={var2}"}],
            "output": "",
        }
    )
    print("\n\n########################")
    print("## here is your pipeline result:")
    print("########################\n")
    print(result.get("output", result))


if __name__ == "__main__":
    main()

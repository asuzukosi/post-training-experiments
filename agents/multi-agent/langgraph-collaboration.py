"""two-agent collaboration graph using create_agent workers."""

from __future__ import annotations

from langchain.tools import tool

from pipeline import build_sequential_graph, make_agent_node



@tool
def search(query: str) -> str:
    """search for facts and statistics."""
    return (
        "stub uk gdp (billions usd): 2019=2718, 2018=2571, 2017=2424, "
        "2016=2248, 2015=2061"
    )


@tool
def run_python(code: str) -> str:
    """execute python for charts or calculations."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, {})
    except Exception as exc:
        return f"error: {exc}"
    return buf.getvalue() or "ok"


def main() -> None:
    researcher = make_agent_node(
        "researcher: gather accurate data for charts. use search. "
        "prefix final answer with FINAL ANSWER when done.",
        tools=[search],
    )
    chart = make_agent_node(
        "chart generator: use run_python to plot data from prior messages.",
        tools=[run_python],
    )
    graph = build_sequential_graph([("researcher", researcher), ("chart", chart)])
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "research uk gdp over 5 years and describe a line chart",
                }
            ],
            "output": "",
        }
    )
    print(result["output"])


if __name__ == "__main__":
    main()

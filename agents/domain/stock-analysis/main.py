"""cli entry for stock analysis langgraph pipeline."""

from __future__ import annotations

from textwrap import dedent



from agent import build_stock_graph


def main() -> None:
    print("## welcome to financial analysis pipeline")
    print("-------------------------------")
    company = input(
        dedent(
            """
      what is the company you want to analyze?
    """
        )
    ).strip()

    graph = build_stock_graph(company)
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": f"analyze {company}"}],
            "output": "",
        }
    )

    print("\n\n########################")
    print("## here is the report")
    print("########################\n")
    print(result.get("output", result))


if __name__ == "__main__":
    main()

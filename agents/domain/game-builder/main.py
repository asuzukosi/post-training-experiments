"""game builder: sequential three-role langgraph pipeline."""

from __future__ import annotations

from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node


SENIOR_ENGINEER_BASE = dedent(
    """
    role: senior software engineer
    goal: create software as needed
    backstory: you are a senior software engineer at a leading tech think tank.
    your expertise is programming in python and you do your best to produce perfect code.
    """
).strip()

QA_ENGINEER_BASE = dedent(
    """
    role: software quality control engineer
    goal: create perfect code by analyzing the code that is given for errors
    backstory: you are a software engineer that specializes in checking code for errors.
    you have an eye for detail and a knack for finding hidden bugs.
    you check for missing imports, variable declarations, mismatched brackets and syntax errors.
    you also check for security vulnerabilities and logic errors.
    """
).strip()

CHIEF_QA_BASE = dedent(
    """
    role: chief software quality control engineer
    goal: ensure that the code does the job that it is supposed to do
    backstory: you feel that programmers always do only half the job, so you are
    super dedicated to making high quality code.
    """
).strip()


def build_graph(game: str):
    code_prompt = dedent(
        f"""
        {SENIOR_ENGINEER_BASE}

        you will create a game using python. these are the instructions:

        instructions
        ------------
        {game}

        your final answer must be the full python code, only the python code and nothing else.
        """
    ).strip()

    review_prompt = dedent(
        f"""
        {QA_ENGINEER_BASE}

        you are helping create a game using python. these are the instructions:

        instructions
        ------------
        {game}

        using the code you got, check for errors. check for logic errors,
        syntax errors, missing imports, variable declarations, mismatched brackets,
        and security vulnerabilities.

        your final answer must be the full python code, only the python code and nothing else.
        """
    ).strip()

    evaluate_prompt = dedent(
        f"""
        {CHIEF_QA_BASE}

        you are helping create a game using python. these are the instructions:

        instructions
        ------------
        {game}

        you will look over the code to ensure that it is complete and
        does the job that it is supposed to do.

        your final answer must be the full python code, only the python code and nothing else.
        """
    ).strip()

    return build_sequential_graph(
        [
            ("senior_engineer", make_agent_node(code_prompt)),
            ("qa_engineer", make_agent_node(review_prompt)),
            ("chief_qa", make_agent_node(evaluate_prompt)),
        ]
    )


def main() -> None:
    print("## Welcome to the Game Crew")
    print("-------------------------------")
    game = input("What is the game you would like to build? What will be the mechanics?\n")

    graph = build_graph(game)
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": game}],
            "output": "",
        }
    )

    print("\n\n########################")
    print("## Here is the result")
    print("########################\n")
    print("final code for the game:")
    print(result.get("output", result))


if __name__ == "__main__":
    main()


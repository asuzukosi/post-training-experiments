from langchain.tools import tool


@tool
def calculate(operation: str) -> str:
    """perform a mathematical calculation, e.g. 200*7 or 5000/2*10."""
    try:
        return str(eval(operation))
    except SyntaxError:
        return "error: invalid syntax in mathematical expression"

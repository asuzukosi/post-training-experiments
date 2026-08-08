"""cli for the data science agent."""

from __future__ import annotations
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from config import get_model_id, get_model_name

app = typer.Typer(
    name="ds-agent",
    help="data science agent cli",
    add_completion=False,
)
console = Console()


def _load_env() -> str:
    # config already loads root + app .env
    return get_model_name()


def _run(query: str) -> None:
    from agent import run_agent_executor

    with console.status("[bold]thinking..."):
        response, meta = run_agent_executor(query)

    if meta and meta != "No function call":
        console.print(Panel(meta.strip(), title="tool call", border_style="dim"))

    console.print(Panel(Markdown(str(response)), title="response", border_style="green"))


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="question for the agent"),
) -> None:
    """run a single question and exit."""
    model = _load_env()
    console.print(f"[dim]model: {model} ({get_model_id()})[/dim]")
    _run(prompt)


def _bye() -> None:
    console.print("[dim]bye[/dim]")
    raise typer.Exit(0)

@app.command()
def chat() -> None:
    """interactive chat loop. type exit or quit to stop."""
    model = _load_env()
    console.print(
        Panel(
            f"data science agent\nmodel: {model}\ntype [bold]exit[/bold] to quit",
            border_style="cyan",
        )
    )

    while True:
        try:
            prompt = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            raise typer.Exit(0)

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            _bye()
        _run(prompt)


if __name__ == "__main__":
    app()

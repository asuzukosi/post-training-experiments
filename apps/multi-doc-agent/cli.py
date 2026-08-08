"""cli for the multi-doc agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config import get_model_id, get_model_name
from vectorstore import ensure_vectorstore

app = typer.Typer(
    name="multi-doc-agent",
    help="multi-document qa agent cli",
    add_completion=False,
)
console = Console()


def _load_env() -> str:
    app_dir = Path(__file__).resolve().parent
    load_dotenv(app_dir / ".env")
    load_dotenv(app_dir.parent.parent / ".env")
    return get_model_name()


def _prepare(docs: Optional[Path]) -> None:
    with console.status("[bold]preparing vector store..."):
        ensure_vectorstore(docs)


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
    docs: Optional[Path] = typer.Option(
        None,
        "--docs",
        help="directory of pdf/docx/txt files to index (rebuilds store)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """run a single question and exit."""
    model = _load_env()
    console.print(f"[dim]model: {model} ({get_model_id()})[/dim]")
    _prepare(docs)
    _run(prompt)


@app.command()
def chat(
    docs: Optional[Path] = typer.Option(
        None,
        "--docs",
        help="directory of pdf/docx/txt files to index (rebuilds store)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """interactive chat loop. type exit or quit to stop."""
    model = _load_env()
    _prepare(docs)
    console.print(
        Panel(
            f"multi-doc agent\nmodel: {model}\ntype [bold]exit[/bold] to quit",
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
            console.print("[dim]bye[/dim]")
            raise typer.Exit(0)

        _run(prompt)


if __name__ == "__main__":
    app()

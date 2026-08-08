"""langchain tools for the job posting pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from langchain.tools import tool

_APP_DIR = Path(__file__).resolve().parent
_JOB_EXAMPLE_PATH = _APP_DIR / "job_description_example.md"


def _serper_search(query: str, max_results: int = 5) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "content-type": "application/json",
    }
    response = requests.post(url, headers=headers, data=payload, timeout=30)
    response.raise_for_status()
    results = response.json().get("organic", [])
    lines: list[str] = []
    for result in results[:max_results]:
        lines.append(
            "\n".join(
                [
                    f"title: {result.get('title', '')}",
                    f"link: {result.get('link', '')}",
                    f"snippet: {result.get('snippet', '')}",
                    "-----------------",
                ]
            )
        )
    return "\n".join(lines) if lines else "no results found."


@tool("search_internet")
def search_internet(query: str) -> str:
    """search the internet for information about a topic and return relevant results."""
    return _serper_search(query)


@tool("search_website")
def search_website(query: str, website: str = "") -> str:
    """search a company website or the web for culture, values, and role-related information."""
    search_query = f"site:{website} {query}" if website else query
    return _serper_search(search_query)


@tool("read_job_description_example")
def read_job_description_example() -> str:
    """read the example job description file for style and structure reference."""
    if not _JOB_EXAMPLE_PATH.is_file():
        return f"example file not found at {_JOB_EXAMPLE_PATH}"
    return _JOB_EXAMPLE_PATH.read_text(encoding="utf-8")


RESEARCH_TOOLS = [search_website, search_internet]
WRITER_REVIEW_TOOLS = [search_website, search_internet, read_job_description_example]

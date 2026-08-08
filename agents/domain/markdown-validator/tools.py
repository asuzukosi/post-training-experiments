"""markdown lint tool for the markdown validator agent."""

from __future__ import annotations

import os
import sys

from langchain.tools import tool
from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException


@tool("markdown_validation_tool")
def markdown_validation_tool(file_path: str) -> str:
    """review a markdown file for syntax / lint errors. pass only the file path."""
    print("validating markdown syntax...", file_path)
    try:
        if not os.path.exists(file_path):
            return "could not validate file. the provided file path does not exist."
        scan_result = PyMarkdownApi().scan_path(file_path.strip())
        return str(scan_result)
    except PyMarkdownApiException as exc:
        print(f"api exception: {exc}", file=sys.stderr)
        return f"api exception: {exc}"

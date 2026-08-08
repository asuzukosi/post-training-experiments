import json
import os

import requests
from langchain.tools import tool


def _serper_search(query: str, n_results: int = 5) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "content-type": "application/json",
    }
    response = requests.post(url, headers=headers, data=payload, timeout=30)
    data = response.json()
    results = data.get("organic", [])
    lines = []
    for result in results[:n_results]:
        try:
            lines.append(
                "\n".join(
                    [
                        f"title: {result['title']}",
                        f"link: {result['link']}",
                        f"snippet: {result['snippet']}",
                        "-----------------",
                    ]
                )
            )
        except KeyError:
            continue
    return f"\nsearch result:\n{chr(10).join(lines)}\n"


@tool
def search_internet(query: str) -> str:
    """search the internet about a topic and return relevant results."""
    return _serper_search(query)


@tool
def search_instagram(query: str) -> str:
    """search for instagram posts about a topic."""
    return _serper_search(f"site:instagram.com {query}")

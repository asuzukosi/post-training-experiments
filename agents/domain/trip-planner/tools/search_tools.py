import json
import os

import requests
from langchain.tools import tool


def _serper_search(query: str, top_n: int = 4) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "content-type": "application/json",
    }
    response = requests.post(url, headers=headers, data=payload, timeout=30)
    data = response.json()
    if "organic" not in data:
        return "no results found; check your serper api key."
    lines = []
    for result in data["organic"][:top_n]:
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
    return "\n".join(lines)


@tool
def search_internet(query: str) -> str:
    """search the internet about a topic and return relevant results."""
    return _serper_search(query)

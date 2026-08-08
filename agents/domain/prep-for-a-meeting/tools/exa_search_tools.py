import json
import os

from exa_py import Exa
from langchain.tools import tool


def _exa() -> Exa:
    return Exa(api_key=os.environ["EXA_API_KEY"])


@tool
def search(query: str) -> str:
    """search for webpages based on the query."""
    results = _exa().search(f"{query}", use_autoprompt=True, num_results=3)
    return str(results)


@tool
def find_similar(url: str) -> str:
    """search for webpages similar to a given url returned from search."""
    results = _exa().find_similar(url, num_results=3)
    return str(results)


@tool
def get_contents(ids: str) -> str:
    """get webpage contents; ids is a json list of ids from search."""
    id_list = json.loads(ids) if ids.strip().startswith("[") else [ids]
    contents = str(_exa().get_contents(id_list))
    parts = [part[:1000] for part in contents.split("URL:")]
    return "\n\n".join(parts)


exa_tools = [search, find_similar, get_contents]

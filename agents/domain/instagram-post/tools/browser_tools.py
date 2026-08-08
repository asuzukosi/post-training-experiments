import json
import os

import requests
from langchain.tools import tool
from unstructured.partition.html import partition_html


@tool
def scrape_and_summarize_website(website: str) -> str:
    """scrape a website and return its main text content (truncated)."""
    url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
    payload = json.dumps({"url": website})
    headers = {"cache-control": "no-cache", "content-type": "application/json"}
    response = requests.post(url, headers=headers, data=payload, timeout=60)
    elements = partition_html(text=response.text)
    content = "\n\n".join(str(el) for el in elements)
    max_len = 8000
    if len(content) > max_len:
        content = content[:max_len] + "\n...(truncated)"
    return f"\nscrapped content:\n{content}\n"

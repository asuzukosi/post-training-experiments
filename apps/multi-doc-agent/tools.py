"""retrieval tools for the multi-doc agent."""

from __future__ import annotations

from langchain.tools import tool
from pydantic import BaseModel, Field
from vectorstore import get_retriever


class RetrieveDocsSchema(BaseModel):
    query: str = Field(description="search query over the indexed documents")


@tool("retrieve-docs", args_schema=RetrieveDocsSchema)
def retrieve_docs(query: str) -> str:
    """retrieve relevant document chunks for a question."""
    # use the vector store retriever to get the relevant document chunks
    docs = get_retriever().invoke(query)
    if not docs:
        return "no relevant documents found"
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={source}\n{doc.page_content}")
    return "\n\n".join(parts)


RETRIEVAL_TOOLS = [retrieve_docs]

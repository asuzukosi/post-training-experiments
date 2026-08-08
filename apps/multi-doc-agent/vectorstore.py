"""chroma vector store helpers for the multi-doc agent."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import PERSIST_DIRECTORY
from documents import chunk_documents, load_documents

_COLLECTION = "multi_doc_agent"
_VECTORSTORE: Chroma | None = None


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def build_vectorstore(
    docs_path: str | Path,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """load docs, chunk, embed, and persist to chroma."""
    global _VECTORSTORE
    persist = Path(persist_directory or PERSIST_DIRECTORY)
    persist.mkdir(parents=True, exist_ok=True)

    documents = load_documents(docs_path) # load documents from the docs path
    if not documents:
        raise ValueError(f"no supported documents found in {docs_path}")

    chunks = chunk_documents(documents) # chunk the documents
    # create vector store from the chunks
    _VECTORSTORE = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=_COLLECTION,
        persist_directory=str(persist),
    )
    return _VECTORSTORE


def load_vectorstore(persist_directory: str | Path | None = None) -> Chroma:
    """load an existing chroma persist directory."""
    global _VECTORSTORE
    persist = Path(persist_directory or PERSIST_DIRECTORY)
    if not persist.exists():
        raise FileNotFoundError(
            f"no vector store at {persist}. run ingest with a --docs path first."
        )
    # load existing vector store from the persist directory
    _VECTORSTORE = Chroma(
        collection_name=_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(persist)
    )
    return _VECTORSTORE


def get_vectorstore() -> Chroma:
    if _VECTORSTORE is None:
        return load_vectorstore()
    return _VECTORSTORE


def get_retriever(k: int = 6):
    return get_vectorstore().as_retriever(search_kwargs={"k": k}) # retreiver is constructed using the vector store


def ensure_vectorstore(docs_path: str | Path | None = None) -> Chroma:
    """build from docs_path if given, otherwise load persisted store."""
    if docs_path:
        return build_vectorstore(docs_path)
    return get_vectorstore()

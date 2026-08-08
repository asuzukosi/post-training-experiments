"""document loading and chunking for the multi-doc agent."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


def load_documents(path: str | Path) -> list[Document]:
    """load pdf, docx, and txt files from a directory."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"docs path not found: {directory}")

    documents: list[Document] = []
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(file_path)).load())
        elif suffix in {".docx", ".doc"}:
            documents.extend(Docx2txtLoader(str(file_path)).load())
        elif suffix == ".txt":
            documents.extend(TextLoader(str(file_path)).load())
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """split documents into overlapping character chunks."""
    # build text splitter to split the documents into overlapping character chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # split the documents into overlapping character chunks
    return splitter.split_documents(documents)

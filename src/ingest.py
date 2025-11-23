import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    BSHTMLLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: str) -> List:
    """
    Load TXT, PDF, and HTML documents from data_dir recursively.
    Returns a list of LangChain Document objects.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    loaders = [
        DirectoryLoader(
            path=str(data_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        ),
        DirectoryLoader(
            path=str(data_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        ),
        DirectoryLoader(
            path=str(data_path),
            glob="**/*.htm*",
            loader_cls=BSHTMLLoader,
            show_progress=True
        )
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    return docs


def clean_text(text: str) -> str:
    """
    Minimal cleaning to normalize whitespace without harming content.
    """
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text.strip()


def split_documents(docs: List, chunk_size: int, chunk_overlap: int) -> List:
    """
    Split documents into overlapping chunks and enrich metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.page_content = clean_text(c.page_content)

        c.metadata = c.metadata or {}
        c.metadata["chunk_index"] = i

        # Normalize stored path
        src = c.metadata.get("source")
        if src:
            c.metadata["source"] = os.path.relpath(src)

    return chunks

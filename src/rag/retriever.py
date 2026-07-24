"""Load a ticker's FAISS index and retrieve relevant 10-K passages."""
from __future__ import annotations
from pathlib import Path

from langchain_community.vectorstores import FAISS

from .ingest import INDEX_DIR, get_embeddings


def has_index(ticker: str) -> bool:
    return (INDEX_DIR / ticker.upper() / "index.faiss").exists()


def retrieve(ticker: str, query: str, k: int = 4) -> list[dict]:
    """Return top-k passages with scores. Empty list if no index."""
    if not has_index(ticker):
        return []
    store = FAISS.load_local(
        str(INDEX_DIR / ticker.upper()), get_embeddings(),
        allow_dangerous_deserialization=True,  # our own local files
    )
    hits = store.similarity_search_with_score(query, k=k)
    return [
        {"text": doc.page_content, "chunk": doc.metadata.get("chunk"),
         "score": float(score)}
        for doc, score in hits
    ]

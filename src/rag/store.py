"""Vector-store abstraction: FAISS or Chroma, chosen at runtime.

Why both: they solve the same problem with different tradeoffs, and swapping
between them is a one-line config change rather than a rewrite.

    FAISS   — Meta's similarity-search library. An in-process index you load
              into memory and save as flat files. Fastest for a fixed corpus,
              no server, but no metadata filtering or incremental upserts.
    Chroma  — a document-oriented vector database. Persists a collection,
              supports metadata filters and add/update/delete on the fly.
              Slightly more overhead; far better when documents change.

Select with the VECTOR_STORE env var (default: faiss):
    VECTOR_STORE=chroma python -m src.rag.ingest --ticker NVDA
"""
from __future__ import annotations
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = Path("data/indexes")          # FAISS: flat files per ticker
CHROMA_DIR = Path("data/chroma")          # Chroma: one persistent client
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None


def backend() -> str:
    """'faiss' or 'chroma' — read per call so tests/CLI can override it."""
    return os.getenv("VECTOR_STORE", "faiss").strip().lower()


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local embedding model, cached process-wide.

    Loading the model costs ~2s and ~90MB, so a module-level cache keeps the
    retrieval step inside the Risk agent's latency budget.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _collection(ticker: str) -> str:
    return f"tenk_{ticker.upper()}"


# ---------- write ----------

def build(ticker: str, chunks: list[str], metadatas: list[dict]) -> str:
    """Embed chunks and persist them. Returns a human-readable location."""
    if backend() == "chroma":
        from langchain_chroma import Chroma

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name=_collection(ticker),
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        # Rebuild cleanly so re-ingesting the same filing can't double up.
        try:
            store.delete_collection()
        except Exception:
            pass
        Chroma.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            metadatas=metadatas,
            collection_name=_collection(ticker),
            persist_directory=str(CHROMA_DIR),
        )
        return f"{CHROMA_DIR}/ (collection: {_collection(ticker)})"

    from langchain_community.vectorstores import FAISS

    store = FAISS.from_texts(chunks, get_embeddings(), metadatas=metadatas)
    out = INDEX_DIR / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    return str(out)


# ---------- read ----------

def exists(ticker: str) -> bool:
    if backend() == "chroma":
        try:
            from langchain_chroma import Chroma

            if not CHROMA_DIR.exists():
                return False
            store = Chroma(
                collection_name=_collection(ticker),
                embedding_function=get_embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
            return store._collection.count() > 0
        except Exception:
            return False
    return (INDEX_DIR / ticker.upper() / "index.faiss").exists()


def search(ticker: str, query: str, k: int) -> list[tuple[str, dict, float]]:
    """Top-k nearest chunks as (text, metadata, distance).

    Both backends return a DISTANCE here (lower = more similar), so the caller
    can convert to a similarity consistently regardless of which is active.
    """
    if backend() == "chroma":
        from langchain_chroma import Chroma

        store = Chroma(
            collection_name=_collection(ticker),
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        hits = store.similarity_search_with_score(query, k=k)
    else:
        from langchain_community.vectorstores import FAISS

        store = FAISS.load_local(
            str(INDEX_DIR / ticker.upper()), get_embeddings(),
            allow_dangerous_deserialization=True,  # our own local files
        )
        hits = store.similarity_search_with_score(query, k=k)

    return [(doc.page_content, doc.metadata, float(score)) for doc, score in hits]

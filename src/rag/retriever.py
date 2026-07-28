"""Query-time half of the RAG pipeline.

    embed(query) -> top-k nearest chunks -> LLM answers from those chunks

Returns telemetry alongside the passages (store, k, latency, similarity) so the
UI can show the retrieval actually happening instead of asserting that it did.
"""
from __future__ import annotations
import time

from . import store

# The Risk agent's standing question against the 10-K.
RISK_QUERY = ("risk factors, competition, regulatory risk, supply chain, "
              "customer concentration, litigation")


def has_index(ticker: str) -> bool:
    return store.exists(ticker)


def _similarity(distance: float) -> float:
    """Distance -> 0..1 similarity for display.

    Embeddings are L2-normalised, so squared L2 distance falls in [0, 4] and
    maps linearly onto a similarity. Presentation only; ranking already
    happened in the vector store.
    """
    return round(max(0.0, min(1.0, 1.0 - distance / 4.0)), 3)


def retrieve(ticker: str, query: str = RISK_QUERY, k: int = 4) -> list[dict]:
    """Top-k passages. Empty list if nothing is indexed for this ticker."""
    if not has_index(ticker):
        return []
    hits = store.search(ticker, query, k)
    return [
        {
            "text": text,
            "chunk": meta.get("chunk"),
            "distance": round(dist, 4),
            "similarity": _similarity(dist),
        }
        for text, meta, dist in hits
    ]


def retrieve_with_telemetry(ticker: str, query: str = RISK_QUERY,
                            k: int = 4) -> dict:
    """retrieve() plus the numbers worth watching live in the UI."""
    t0 = time.time()
    passages = retrieve(ticker, query, k)
    return {
        "passages": passages,
        "store": store.backend(),
        "embed_model": store.EMBED_MODEL.split("/")[-1],
        "k": k,
        "query": query,
        "hits": len(passages),
        "latency_ms": int((time.time() - t0) * 1000),
        "grounded": bool(passages),
    }

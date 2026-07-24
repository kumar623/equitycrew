"""EquityCrew API — FastAPI backend with live agent-progress streaming.

Run:   uvicorn api:app --reload
Open:  http://127.0.0.1:8000
"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from src.graph import build_graph

app = FastAPI(title="EquityCrew", version="1.0")
STATIC = Path(__file__).parent / "static"

# labels shown in the UI as each node runs
NODE_LABELS = {
    "financials": "Financials agent — fetching live price & fundamentals",
    "news": "News agent — reading recent headlines",
    "risk": "Risk agent — analyzing risks (10-K RAG if indexed)",
    "writer": "Writer agent — drafting the memo",
    "critic": "Critic agent — reviewing the draft",
    "verifier": "Verifier agent — fact-checking every number vs fresh data",
    "finalize": "Finalizing memo",
}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/research/stream")
def research_stream(ticker: str):
    """Server-Sent Events: one event per agent step, then the final memo."""
    def gen():
        try:
            graph = build_graph()
            state: dict = {}
            for event in graph.stream(
                {"ticker": ticker.upper(), "revision_count": 0},
                stream_mode="updates",
            ):
                for node, update in event.items():
                    state.update(update or {})
                    payload = {
                        "node": node,
                        "label": NODE_LABELS.get(node, node),
                        "approved": state.get("approved"),
                        "revision_count": state.get("revision_count", 0),
                    }
                    if node == "verifier":
                        payload["verification"] = state.get("verification")
                    if node == "finalize":
                        payload["memo"] = state.get("final_memo")
                    yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'node': 'done'})}\n\n"
        except Exception as e:  # surface errors to the UI instead of dying silently
            yield f"data: {json.dumps({'node': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

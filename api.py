"""EquityCrew API — FastAPI backend with live agent-progress streaming.

Run:   uvicorn api:app --reload
Open:  http://127.0.0.1:8000
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from src.graph import build_graph
from src.tools.market_data import resolve_ticker
from src.usage import UsageTracker

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


def _detail(node: str, state: dict) -> dict:
    """What this agent actually produced — surfaced in the UI so the pipeline
    is inspectable rather than a row of checkmarks."""
    if node == "financials":
        f = state.get("financials", {})
        return {"summary": f.get("summary"), "data": f.get("data", {})}
    if node == "news":
        n = state.get("news", {})
        headlines = (n.get("data") or {}).get("headlines", [])
        return {"summary": n.get("summary"),
                "headlines": [h.get("title") for h in headlines][:6]}
    if node == "risk":
        risks = state.get("risks") or ""
        # the risk agent cites "[chunk 12]" only when grounded in a real filing
        return {"summary": risks, "rag_grounded": "[chunk" in risks}
    if node == "writer":
        draft = state.get("draft") or ""
        return {"words": len(draft.split())}
    if node == "critic":
        return {"approved": state.get("approved"),
                "feedback": state.get("critique")}
    if node == "verifier":
        return state.get("verification") or {}
    if node == "finalize":
        return {"memo": state.get("final_memo")}
    return {}


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
            # Preflight: a symbol with no market data would otherwise burn a
            # full crew run (~60s, ~$0.10) to conclude it has nothing to say.
            check = resolve_ticker(ticker)
            if not check["ok"]:
                yield f"data: {json.dumps({'node': 'invalid_ticker', **check})}\n\n"
                return

            graph = build_graph()
            tracker = UsageTracker()
            state: dict = {}
            t0 = time.time()
            last = t0
            for event in graph.stream(
                {"ticker": ticker.upper(), "revision_count": 0},
                stream_mode="updates",
                config={"callbacks": [tracker]},
            ):
                for node, update in event.items():
                    state.update(update or {})
                    now = time.time()
                    payload = {
                        "node": node,
                        "label": NODE_LABELS.get(node, node),
                        "approved": state.get("approved"),
                        "revision_count": state.get("revision_count", 0),
                        "detail": _detail(node, state),
                        "took_s": round(now - last, 1),
                        "elapsed_s": round(now - t0, 1),
                        "usage": tracker.snapshot(),
                    }
                    last = now
                    # kept top-level for backwards compatibility with the UI
                    if node == "verifier":
                        payload["verification"] = state.get("verification")
                    if node == "finalize":
                        payload["memo"] = state.get("final_memo")
                    yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'node': 'done', 'usage': tracker.snapshot(), 'elapsed_s': round(time.time() - t0, 1)})}\n\n"
        except Exception as e:  # surface errors to the UI instead of dying silently
            yield f"data: {json.dumps({'node': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

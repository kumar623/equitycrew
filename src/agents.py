"""The agent crew. Each function is a node in the LangGraph graph.

Design rules:
- Numbers come ONLY from tools (market_data), never from the model.
- Each agent has a tight, single responsibility.
- The Critic can send the draft back to the Writer once (loop is capped).
"""
from __future__ import annotations
import json

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .config import get_llm, MAX_REVISIONS
from .state import ResearchState
from .tools.market_data import get_price, get_fundamentals, get_news

class _LazyLLM:
    """Defers creating the Anthropic client until first use, so the app can
    import/start (e.g. FastAPI boot, graph wiring) without a key present."""
    _inner = None

    def __getattr__(self, name):
        if _LazyLLM._inner is None:
            _LazyLLM._inner = get_llm()
        return getattr(_LazyLLM._inner, name)


llm = _LazyLLM()


def _text(response) -> str:
    """claude-sonnet-5 thinks adaptively, so .content may be a list of blocks
    (thinking + text) instead of a string. Always reduce to plain text —
    a thinking block fed back into a user message is a 400."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


# ---------- research agents (call tools, summarize) ----------

def financials_agent(state: ResearchState) -> ResearchState:
    ticker = state["ticker"]
    data = {"price": get_price(ticker), "fundamentals": get_fundamentals(ticker)}
    msg = [
        SystemMessage(content=(
            "You are a financials analyst. Summarize the company's fundamentals in "
            "3-4 sentences. Use ONLY the numbers provided. Do not invent figures."
        )),
        HumanMessage(content=f"Data:\n{json.dumps(data, indent=2)}"),
    ]
    summary = _text(llm.invoke(msg))
    return {"financials": {"data": data, "summary": summary}}


def news_agent(state: ResearchState) -> ResearchState:
    ticker = state["ticker"]
    data = get_news(ticker)
    msg = [
        SystemMessage(content=(
            "You are a news & sentiment analyst. From these headlines, give a 2-3 "
            "sentence read on recent sentiment and any catalysts. If no headlines, say so."
        )),
        HumanMessage(content=f"Headlines:\n{json.dumps(data, indent=2)}"),
    ]
    summary = _text(llm.invoke(msg))
    return {"news": {"data": data, "summary": summary}}


def risk_agent(state: ResearchState) -> ResearchState:
    """RAG-grounded when a 10-K index exists (src.rag.ingest), else falls back
    to reasoning over live fundamentals."""
    ticker = state["ticker"]

    passages = []
    try:
        from .rag.retriever import retrieve
        passages = retrieve(
            ticker,
            "risk factors, competition, regulatory risk, supply chain, "
            "customer concentration, litigation",
            k=4,
        )
    except Exception:  # RAG deps not installed or no index — fall back gracefully
        passages = []

    if passages:
        context = "\n\n---\n\n".join(
            f"[10-K chunk {p['chunk']}]\n{p['text'][:1500]}" for p in passages
        )
        msg = [
            SystemMessage(content=(
                "You are a risk analyst. Using ONLY these excerpts from the company's "
                "10-K, summarize the 3-4 most important risks in plain language. "
                "Cite chunks like [chunk 12]. Do not add risks not in the excerpts."
            )),
            HumanMessage(content=f"{ticker} 10-K excerpts:\n{context}"),
        ]
    else:
        fund = state.get("financials", {}).get("data", {})
        msg = [
            SystemMessage(content=(
                "You are a risk analyst. In 2-3 sentences, list the key risks/red flags "
                "given the fundamentals. Be specific and grounded; no boilerplate. "
                "Note that no 10-K filing was available for citation."
            )),
            HumanMessage(content=f"{ticker} fundamentals:\n{json.dumps(fund, indent=2)}"),
        ]
    return {"risks": _text(llm.invoke(msg))}


# ---------- writer ----------

def writer_agent(state: ResearchState) -> ResearchState:
    prior_critique = state.get("critique")
    revise_note = (
        f"\n\nA reviewer asked you to revise. Address this feedback:\n{prior_critique}"
        if prior_critique else ""
    )
    context = {
        "ticker": state["ticker"],
        "financials": state.get("financials", {}).get("summary"),
        "news": state.get("news", {}).get("summary"),
        "risks": state.get("risks"),
    }
    msg = [
        SystemMessage(content=(
            "You are an equity research writer. Produce a concise investment memo with "
            "sections: Thesis, Fundamentals, News & Sentiment, Key Risks, Bull vs Bear, "
            "and a Rating (Buy/Hold/Sell). Ground every claim in the provided material. "
            "End with: 'This is not investment advice.'" + revise_note
        )),
        HumanMessage(content=json.dumps(context, indent=2)),
    ]
    return {"draft": _text(llm.invoke(msg))}


# ---------- critic (structured decision + loop control) ----------

class Critique(BaseModel):
    approved: bool = Field(description="True if the memo is well-supported and complete.")
    feedback: str = Field(description="Specific issues to fix if not approved.")


def critic_agent(state: ResearchState) -> ResearchState:
    structured = llm.with_structured_output(Critique)
    msg = [
        SystemMessage(content=(
            "You are a senior reviewer. Check the memo for unsupported claims, missing "
            "sections, or invented numbers. Approve only if solid."
        )),
        HumanMessage(content=state["draft"]),
    ]
    result: Critique = structured.invoke(msg)
    count = state.get("revision_count", 0)
    return {
        "approved": result.approved,
        "critique": result.feedback,
        "revision_count": count + (0 if result.approved else 1),
    }


def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: approve -> finalize, else revise (until cap)."""
    if state.get("approved") or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "finalize"
    return "revise"


# ---------- verifier (self-correcting numeric check) ----------

class Verification(BaseModel):
    all_numbers_correct: bool = Field(
        description="True if every figure in the memo matches the fresh data.")
    corrected_memo: str = Field(
        description="The memo with any incorrect figures fixed. If all correct, "
                    "return the memo unchanged.")
    mismatches: list[str] = Field(
        default_factory=list,
        description="List of figures that were wrong and how they were fixed.")


def verifier_agent(state: ResearchState) -> ResearchState:
    """Re-fetch live data and check every number in the approved draft against it.
    This is the 'run the tests and fix failures' loop, applied to research."""
    ticker = state["ticker"]
    fresh = {
        "price": get_price(ticker),
        "fundamentals": get_fundamentals(ticker),
    }
    structured = llm.with_structured_output(Verification)
    msg = [
        SystemMessage(content=(
            "You are a fact-checker. Compare every numeric claim in the memo against "
            "the FRESH data provided (the single source of truth). Fix any figure "
            "that does not match. Do not change analysis or wording otherwise."
        )),
        HumanMessage(content=(
            f"FRESH DATA:\n{json.dumps(fresh, indent=2)}\n\nMEMO:\n{state['draft']}"
        )),
    ]
    result: Verification = structured.invoke(msg)
    # Guard: a truncated/malformed structured response must never replace a
    # full memo with a fragment — keep the original draft in that case.
    corrected = result.corrected_memo or ""
    if len(corrected) < 0.5 * len(state["draft"]):
        corrected = state["draft"]
    return {
        "draft": corrected,
        "verification": {
            "all_correct": result.all_numbers_correct,
            "mismatches": result.mismatches,
        },
    }


def finalize(state: ResearchState) -> ResearchState:
    return {"final_memo": state["draft"]}

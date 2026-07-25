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

from .config import AGENT_MODELS, get_llm, DISABLE_RAG, MAX_REVISIONS, MODEL_NAME
from .state import ResearchState
from .tools.market_data import get_price, get_fundamentals, get_news

_clients: dict = {}


def llm_for(agent: str):
    """Client for one agent's configured model, built on first use so the app
    can import and boot (FastAPI, graph wiring) without a key present.
    Clients are cached per model, so agents sharing a model share a client."""
    model = AGENT_MODELS.get(agent, MODEL_NAME)
    if model not in _clients:
        _clients[model] = get_llm(model)
    return _clients[model]


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
    summary = _text(llm_for("financials").invoke(msg))
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
    summary = _text(llm_for("news").invoke(msg))
    return {"news": {"data": data, "summary": summary}}


def risk_agent(state: ResearchState) -> ResearchState:
    """RAG-grounded when a 10-K index exists (src.rag.ingest), else falls back
    to reasoning over live fundamentals."""
    ticker = state["ticker"]

    passages = []
    try:
        if DISABLE_RAG:
            raise RuntimeError("RAG disabled by EQUITYCREW_DISABLE_RAG")
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
    return {"risks": _text(llm_for("risk").invoke(msg))}


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
    return {"draft": _text(llm_for("writer").invoke(msg))}


# ---------- critic (structured decision + loop control) ----------

class Critique(BaseModel):
    approved: bool = Field(description="True if the memo is well-supported and complete.")
    feedback: str = Field(description="Specific issues to fix if not approved.")


def critic_agent(state: ResearchState) -> ResearchState:
    structured = llm_for("critic").with_structured_output(Critique)
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

class Correction(BaseModel):
    find: str = Field(
        description="The exact incorrect text as it appears in the memo, copied "
                    "character for character (e.g. '$210.77' or '32.28x').")
    replace: str = Field(
        description="What that text should say according to the fresh data.")
    note: str = Field(description="Short human-readable description of the fix.")


class Verification(BaseModel):
    all_numbers_correct: bool = Field(
        description="True if every figure in the memo matches the fresh data.")
    corrections: list[Correction] = Field(
        default_factory=list,
        description="One entry per figure that disagrees with the fresh data. "
                    "Return an empty list if everything matches.")


def verifier_agent(state: ResearchState) -> ResearchState:
    """Re-fetch live data and check every number in the approved draft against it.
    This is the 'run the tests and fix failures' loop, applied to research."""
    ticker = state["ticker"]
    fresh = {
        "price": get_price(ticker),
        "fundamentals": get_fundamentals(ticker),
    }
    structured = llm_for("verifier").with_structured_output(Verification)
    msg = [
        SystemMessage(content=(
            "You are a fact-checker. Compare every numeric claim in the memo against "
            "the FRESH data provided (the single source of truth). For each figure "
            "that does not match, return a correction whose 'find' is the exact text "
            "as it appears in the memo. Do not rewrite the memo and do not change "
            "analysis or wording — return corrections only."
        )),
        HumanMessage(content=(
            f"FRESH DATA:\n{json.dumps(fresh, indent=2)}\n\nMEMO:\n{state['draft']}"
        )),
    ]
    result: Verification = structured.invoke(msg)

    # Apply the patches here rather than having the model re-emit the memo:
    # ~1,400 fewer output tokens per run, and a malformed response can no
    # longer truncate or destroy the draft — the worst case is no edit.
    draft = state["draft"]
    applied, unapplied = [], []
    for c in result.corrections:
        if c.find and c.find in draft:
            draft = draft.replace(c.find, c.replace, 1)
            applied.append(f"{c.note} ({c.find} → {c.replace})")
        elif c.find:
            unapplied.append(f"{c.note} (could not locate “{c.find}” in the memo)")

    return {
        "draft": draft,
        "verification": {
            "all_correct": result.all_numbers_correct and not applied,
            "mismatches": applied,
            "unapplied": unapplied,
        },
    }


def finalize(state: ResearchState) -> ResearchState:
    return {"final_memo": state["draft"]}

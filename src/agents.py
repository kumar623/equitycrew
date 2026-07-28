"""The agent crew. Each function is a node in the LangGraph graph.

Design rules:
- Numbers come ONLY from tools (market_data), never from the model.
- Each agent has a tight, single responsibility.
- The Critic can send the draft back to the Writer once (loop is capped).
"""
from __future__ import annotations
import json
import re

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

    passages: list = []
    telemetry: dict = {"grounded": False, "passages": []}
    try:
        if DISABLE_RAG:
            raise RuntimeError("RAG disabled by EQUITYCREW_DISABLE_RAG")
        from .rag.retriever import retrieve_with_telemetry
        telemetry = retrieve_with_telemetry(ticker, k=4)
        passages = telemetry["passages"]
    except Exception as e:  # deps missing or no index — degrade, don't crash
        telemetry = {"grounded": False, "passages": [], "error": str(e)[:120]}
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
        # Fetch rather than read from state: the research agents run in
        # parallel, so financials may not have landed yet. One cheap tool call
        # buys independence, and this path only runs when no 10-K is indexed.
        fund = get_fundamentals(ticker)
        msg = [
            SystemMessage(content=(
                "You are a risk analyst. In 2-3 sentences, list the key risks/red flags "
                "given the fundamentals. Be specific and grounded; no boilerplate. "
                "Note that no 10-K filing was available for citation."
            )),
            HumanMessage(content=f"{ticker} fundamentals:\n{json.dumps(fund, indent=2)}"),
        ]
    return {"risks": _text(llm_for("risk").invoke(msg)), "retrieval": telemetry}


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


#: (label, pattern that must appear somewhere in the memo)
REQUIRED_SECTIONS = [
    ("Thesis", r"thesis"),
    ("Fundamentals", r"fundamental"),
    ("News & Sentiment", r"news|sentiment"),
    ("Key Risks", r"risk"),
    ("Bull vs Bear", r"bull"),
    ("Rating", r"\b(buy|hold|sell)\b"),
]


def structural_defects(memo: str) -> list:
    """Deterministic completeness check.

    Whether a section exists is a fact, not a judgement, and an LLM asked to
    check it is unreliable — it read a memo with the rating stripped out and
    approved it anyway. Checking in code makes that class of defect impossible
    to miss, and saves an LLM call when the draft is obviously broken.
    """
    low = (memo or "").lower()
    missing = [label for label, pat in REQUIRED_SECTIONS if not re.search(pat, low)]
    defects = [f"Missing required section: {m}" for m in missing]
    if len((memo or "").split()) < 150:
        defects.append("Memo is too short to be complete — likely truncated.")
    return defects


def critic_agent(state: ResearchState) -> ResearchState:
    # Structure first: cheap, certain, and no API call when it fails.
    defects = structural_defects(state.get("draft", ""))
    if defects:
        count = state.get("revision_count", 0)
        return {
            "approved": False,
            "critique": "The draft is structurally incomplete. "
                        + " ".join(defects)
                        + " Rewrite the memo with every required section present.",
            "revision_count": count + 1,
        }

    structured = llm_for("critic").with_structured_output(Critique)
    # The critic used to see the draft alone. It therefore could not tell a
    # grounded figure from an invented one, and rejected almost every memo
    # demanding citations for numbers that came from live tool calls. Giving
    # it the research makes "unsupported claim" an answerable question.
    research = {
        "financials": state.get("financials", {}).get("summary"),
        "news": state.get("news", {}).get("summary"),
        "risks": state.get("risks"),
    }
    msg = [
        SystemMessage(content=(
            "You are a senior reviewer deciding whether an investment memo is fit to "
            "publish. You are given the research the writer worked from, and the draft.\n\n"
            "Context you need: every figure in the research came from live market-data "
            "tool calls made by this system, and a separate verifier re-checks each "
            "number against a fresh data pull after you. So do NOT ask for inline "
            "citations, sources or as-of dates for figures — that is already handled.\n\n"
            "First, check completeness mechanically. The memo must contain all six "
            "sections — thesis, fundamentals, news/sentiment, key risks, bull vs bear, "
            "and a rating — and the rating must state an explicit Buy, Hold or Sell. "
            "If any section is absent, or no explicit Buy/Hold/Sell verdict appears, "
            "reject. Do not infer a missing section from related prose elsewhere.\n\n"
            "Then reject for any other substantive defect:\n"
            "  - a figure or fact appears in the memo but not in the research\n"
            "  - a claim contradicts the research or another part of the memo\n"
            "  - the rating is unsupported by the reasoning given\n"
            "  - the memo is truncated or structurally broken\n\n"
            "If the memo is complete, internally consistent and grounded in the "
            "research, approve it. Do not withhold approval over style, tone, hedging, "
            "or extra caveats you would have worded differently."
        )),
        HumanMessage(content=(
            f"RESEARCH THE WRITER WAS GIVEN:\n{json.dumps(research, indent=2)}\n\n"
            f"DRAFT MEMO:\n{state['draft']}"
        )),
    ]
    result: Critique = structured.invoke(msg)
    count = state.get("revision_count", 0)
    return {
        "approved": result.approved,
        "critique": result.feedback,
        "revision_count": count + (0 if result.approved else 1),
    }


def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: approve -> finalize, else revise (until cap).

    The comparison is `>` and not `>=` on purpose. The critic has already
    incremented revision_count for the rejection being routed on, so `>=`
    would treat the very first rejection as having exhausted the budget and
    skip the rewrite entirely — MAX_REVISIONS=1 would permit zero revisions.
    """
    if state.get("approved") or state.get("revision_count", 0) > MAX_REVISIONS:
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

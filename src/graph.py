"""Wire the agents into a LangGraph state graph.

Flow:
    START
      -> financials -> news -> risk   (research; risk uses RAG when 10-K indexed)
      -> writer
      -> critic --(approve or max revisions)--> verifier -> finalize -> END
                 \--(revise)-------------------> writer   (loop, capped)

The verifier re-checks every number in the memo against FRESH tool calls and
fixes mismatches — a self-correcting step before anything is finalized.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import ResearchState
from .agents import (
    financials_agent, news_agent, risk_agent,
    writer_agent, critic_agent, route_after_critic,
    verifier_agent, finalize,
)


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("financials", financials_agent)
    g.add_node("news", news_agent)
    g.add_node("risk", risk_agent)
    g.add_node("writer", writer_agent)
    g.add_node("critic", critic_agent)
    g.add_node("verifier", verifier_agent)
    g.add_node("finalize", finalize)

    g.add_edge(START, "financials")
    g.add_edge("financials", "news")
    g.add_edge("news", "risk")
    g.add_edge("risk", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic", route_after_critic,
        {"revise": "writer", "finalize": "verifier"},
    )
    g.add_edge("verifier", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# module-level compiled graph for reuse (API, MCP, etc.)
graph = build_graph()


def run(ticker: str) -> str:
    result = graph.invoke({"ticker": ticker, "revision_count": 0})
    return result.get("final_memo") or result.get("draft", "(no memo produced)")

"""Shared state passed between agents in the LangGraph graph."""
from __future__ import annotations
from typing import Optional, TypedDict


class ResearchState(TypedDict, total=False):
    ticker: str                 # input
    financials: dict            # Financials agent output
    news: dict                  # News/Sentiment agent output
    risks: str                  # Risk agent output
    draft: str                  # Writer agent output (the memo)
    critique: str               # Critic agent feedback
    approved: bool              # Critic decision
    revision_count: int         # guards the critic->writer loop
    verification: dict          # Verifier agent report (mismatches fixed)
    final_memo: Optional[str]   # assembled result

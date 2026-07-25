"""Token/cost accounting shared by the API and the eval harness."""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from .config import PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK


class UsageTracker(BaseCallbackHandler):
    """Sums token usage across every LLM call in a graph run."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.calls += 1

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens / 1e6 * PRICE_IN_PER_MTOK
                + self.output_tokens / 1e6 * PRICE_OUT_PER_MTOK)

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }

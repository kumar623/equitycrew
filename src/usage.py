"""Token/cost accounting shared by the API and the eval harness."""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from .config import MODEL_PRICES, PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK


class UsageTracker(BaseCallbackHandler):
    """Sums token usage across every LLM call in a graph run."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.cost_usd_acc = 0.0   # priced per call, since models differ per agent

    def on_llm_end(self, response, **kwargs):
        # Which model answered — agents may run on different tiers.
        model = ((response.llm_output or {}).get("model_name")
                 or (response.llm_output or {}).get("model") or "")
        price_in, price_out = MODEL_PRICES.get(
            model, (PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK))
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                tin = usage.get("input_tokens", 0)
                tout = usage.get("output_tokens", 0)
                self.input_tokens += tin
                self.output_tokens += tout
                self.cost_usd_acc += tin / 1e6 * price_in + tout / 1e6 * price_out
                self.calls += 1

    @property
    def cost_usd(self) -> float:
        return self.cost_usd_acc

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }

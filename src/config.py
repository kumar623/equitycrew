"""Central config: loads env vars and builds the shared LLM client."""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# One place to pick the model + settings for every agent.
MODEL_NAME = "claude-sonnet-5"
FAST_MODEL = "claude-haiku-4-5"   # for agents that only summarise tool output
MAX_REVISIONS = 1  # cap the critic->writer loop so it always terminates

# Per-agent model routing. Summarising a JSON blob and writing an investment
# thesis are not the same job, so they need not run on the same model. Each is
# env-overridable (EQUITYCREW_MODEL_NEWS=claude-haiku-4-5) for A/B testing
# without a code change.
AGENT_MODELS = {
    agent: os.getenv(f"EQUITYCREW_MODEL_{agent.upper()}", default)
    for agent, default in {
        # Summarising a JSON blob or five headlines is not a frontier-model job;
        # measured at ~15% of spend combined, Haiku cuts that by two thirds.
        "financials": FAST_MODEL,
        "news": FAST_MODEL,
        # Judgement and drafting stay on the stronger model: the writer produces
        # the deliverable, the critic is the quality gate, and the verifier is
        # the last line of defence against a wrong number reaching the reader.
        "risk": MODEL_NAME,
        "writer": MODEL_NAME,
        "critic": MODEL_NAME,
        "verifier": MODEL_NAME,
    }.items()
}

# $/MTok (input, output) per model — used for per-agent cost attribution.
MODEL_PRICES = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
}

# claude-sonnet-5 list price, USD per million tokens (input, output).
PRICE_IN_PER_MTOK = 3.00
PRICE_OUT_PER_MTOK = 15.00

# Escape hatch for memory-constrained hosts (e.g. a 512MB free instance):
# skips loading the embedding model, so the Risk agent uses its fallback path.
DISABLE_RAG = os.getenv("EQUITYCREW_DISABLE_RAG", "").lower() in ("1", "true", "yes")


def get_llm(model: str = None) -> ChatAnthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # claude-sonnet-5 rejects sampling params (temperature/top_p/top_k) with a 400
    return ChatAnthropic(
        model=model or MODEL_NAME,
        max_tokens=4000,
        api_key=ANTHROPIC_API_KEY,
    )

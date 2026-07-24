"""Central config: loads env vars and builds the shared LLM client."""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# One place to pick the model + settings for every agent.
MODEL_NAME = "claude-sonnet-5"
MAX_REVISIONS = 1  # cap the critic->writer loop so it always terminates


def get_llm() -> ChatAnthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # claude-sonnet-5 rejects sampling params (temperature/top_p/top_k) with a 400
    return ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=4000,
        api_key=ANTHROPIC_API_KEY,
    )

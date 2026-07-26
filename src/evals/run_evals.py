"""M4 eval harness — the numbers that go in the README.

Per ticker:
  1. Full pipeline run    -> numeric accuracy, latency, tokens, $ cost
  2. Verifier catch-rate  -> plant numeric errors in the memo, count fixes
  3. Critic catch-rate    -> feed structurally-broken memos, count rejections

Run:  python -m src.evals.run_evals NVDA [AAPL ...]
Writes data/evals/report-<timestamp>.json and prints a summary.
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..agents import critic_agent, verifier_agent
from ..config import PRICE_IN_PER_MTOK as PRICE_IN, PRICE_OUT_PER_MTOK as PRICE_OUT
from ..graph import build_graph
from ..tools.market_data import get_fundamentals, get_price
from ..usage import UsageTracker
from .metrics import critic_sabotage_variants, numeric_accuracy, perturb_numbers


def eval_ticker(ticker: str) -> dict:
    console = Console()
    console.print(f"\n[bold]== {ticker} ==[/bold]")

    # 1. full pipeline run
    tracker = UsageTracker()
    t0 = time.time()
    state = build_graph().invoke(
        {"ticker": ticker, "revision_count": 0},
        config={"callbacks": [tracker]},
    )
    latency = time.time() - t0
    memo = state["final_memo"]

    # ground truth = data captured in-run + a fresh fetch (price moves between calls)
    run_data = state.get("financials", {}).get("data", {})
    accuracy, ungrounded = numeric_accuracy(
        memo,
        run_data.get("price", {}), run_data.get("fundamentals", {}),
        get_price(ticker), get_fundamentals(ticker),
    )
    console.print(f"pipeline: {latency:.0f}s, {tracker.calls} LLM calls, ${tracker.cost_usd:.3f}")
    console.print(f"numeric accuracy: {accuracy:.0%}"
                  + (f"  (ungrounded: {ungrounded})" if ungrounded else ""))

    # 2. verifier catch-rate: plant numeric errors, expect the verifier to fix them
    sabotaged, planted = perturb_numbers(memo, count=4)
    caught = 0
    if planted:
        v_state = verifier_agent({"ticker": ticker, "draft": sabotaged})
        corrected = v_state["draft"]
        caught = sum(1 for _orig, bad in planted if bad not in corrected)
    verifier_catch = caught / len(planted) if planted else None
    console.print(f"verifier catch-rate: {caught}/{len(planted)} planted errors fixed")

    # 3. critic catch-rate: broken memos should be rejected, the real one approved.
    # Pass the research the critic sees in the real pipeline; without it the
    # critic cannot distinguish a grounded figure from an invented one.
    research_ctx = {
        "financials": state.get("financials"),
        "news": state.get("news"),
        "risks": state.get("risks"),
    }
    rejected, variants = 0, critic_sabotage_variants(memo)
    for name, bad_memo in variants.items():
        try:
            result = critic_agent({"draft": bad_memo, "revision_count": 0, **research_ctx})
        except Exception as e:  # a bad variant must not sink the whole (paid) run
            console.print(f"critic vs {name}: ERROR ({e})")
            continue
        if not result["approved"]:
            rejected += 1
        console.print(f"critic vs {name}: {'rejected ✓' if not result['approved'] else 'APPROVED ✗'}")
    clean = critic_agent({"draft": memo, "revision_count": 0, **research_ctx})
    console.print(f"critic vs clean memo: {'approved ✓' if clean['approved'] else 'REJECTED ✗'}")

    return {
        "ticker": ticker,
        "numeric_accuracy": round(accuracy, 3),
        "ungrounded_numbers": ungrounded,
        "verifier_catch_rate": verifier_catch,
        "critic_catch_rate": round(rejected / len(variants), 3) if variants else None,
        "critic_approves_clean": bool(clean["approved"]),
        "revisions_in_run": state.get("revision_count", 0),
        "latency_s": round(latency, 1),
        "llm_calls": tracker.calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "cost_usd": round(tracker.cost_usd, 4),
        "memo": memo,
    }


def main(tickers: list) -> None:
    console = Console()
    results = [eval_ticker(t.upper()) for t in tickers]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "claude-sonnet-5",
        "pricing_usd_per_mtok": {"input": PRICE_IN, "output": PRICE_OUT},
        "results": results,
    }
    out_dir = Path("data/evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    table = Table(title="EquityCrew evals")
    for col in ("ticker", "numeric acc", "verifier catch", "critic catch",
                "latency", "cost/memo"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r["ticker"],
            f"{r['numeric_accuracy']:.0%}",
            f"{r['verifier_catch_rate']:.0%}" if r["verifier_catch_rate"] is not None else "n/a",
            f"{r['critic_catch_rate']:.0%}" if r["critic_catch_rate"] is not None else "n/a",
            f"{r['latency_s']:.0f}s",
            f"${r['cost_usd']:.3f}",
        )
    console.print(table)
    console.print(f"report: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["NVDA"])

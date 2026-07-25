"""Demo-reliability check — the PRD's "10 consecutive runs succeed" metric.

Runs the full crew N times and records, per run: success, latency, cost, and
whether the memo has all six required sections. A run counts as a failure if
it raises, returns no memo, or is missing a section.

Run:  python -m src.evals.reliability --runs 10 --ticker NVDA
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..graph import build_graph
from ..usage import UsageTracker

REQUIRED_SECTIONS = ["thesis", "fundamental", "news", "risk", "bull", "rating"]


def one_run(ticker: str) -> dict:
    tracker = UsageTracker()
    t0 = time.time()
    try:
        state = build_graph().invoke(
            {"ticker": ticker, "revision_count": 0},
            config={"callbacks": [tracker]},
        )
        memo = state.get("final_memo") or ""
        low = memo.lower()
        missing = [s for s in REQUIRED_SECTIONS if s not in low]
        return {
            "ok": bool(memo) and not missing,
            "missing_sections": missing,
            "chars": len(memo),
            "revisions": state.get("revision_count", 0),
            "latency_s": round(time.time() - t0, 1),
            "cost_usd": round(tracker.cost_usd, 4),
            "error": None,
        }
    except Exception as e:  # a crash is a failed run, not a crashed harness
        return {
            "ok": False, "missing_sections": [], "chars": 0, "revisions": 0,
            "latency_s": round(time.time() - t0, 1), "cost_usd": round(tracker.cost_usd, 4),
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--ticker", default="NVDA")
    args = ap.parse_args()
    console = Console()

    results = []
    for i in range(1, args.runs + 1):
        r = one_run(args.ticker)
        results.append(r)
        console.print(
            f"run {i:>2}/{args.runs}: "
            + ("[green]ok[/green]" if r["ok"] else f"[red]FAIL[/red] {r['error'] or r['missing_sections']}")
            + f"  {r['latency_s']}s  ${r['cost_usd']:.3f}  rev={r['revisions']}"
        )

    passed = sum(r["ok"] for r in results)
    lat = [r["latency_s"] for r in results]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": args.ticker,
        "runs": args.runs,
        "passed": passed,
        "success_rate": round(passed / args.runs, 3),
        "latency_s": {"min": min(lat), "max": max(lat),
                      "mean": round(sum(lat) / len(lat), 1)},
        "cost_usd_total": round(sum(r["cost_usd"] for r in results), 4),
        "cost_usd_mean": round(sum(r["cost_usd"] for r in results) / args.runs, 4),
        "results": results,
    }
    out = Path("data/evals")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"reliability-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))

    t = Table(title=f"Reliability — {passed}/{args.runs} runs succeeded")
    for c in ("metric", "value"):
        t.add_column(c)
    t.add_row("success rate", f"{report['success_rate']:.0%}")
    t.add_row("latency (mean)", f"{report['latency_s']['mean']}s")
    t.add_row("latency (min/max)", f"{report['latency_s']['min']}s / {report['latency_s']['max']}s")
    t.add_row("cost per memo (mean)", f"${report['cost_usd_mean']:.3f}")
    t.add_row("total spend", f"${report['cost_usd_total']:.2f}")
    console.print(t)
    console.print(f"report: {path}")


if __name__ == "__main__":
    main()

"""CLI entry point: python main.py NVDA"""
import sys
from rich.console import Console
from rich.markdown import Markdown

from src.graph import run

console = Console()


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    console.rule(f"[bold]EquityCrew — researching {ticker.upper()}")
    with console.status("Agents at work (financials → news → risk → writer → critic)…"):
        memo = run(ticker)
    console.print(Markdown(memo))


if __name__ == "__main__":
    main()

"""MCP server exposing EquityCrew's tools to any MCP client (e.g. Claude Desktop).

Run:  python mcp_server.py
Then add it to your MCP client config. This lets the same tools that power the
multi-agent graph also be called directly by an LLM over the Model Context Protocol.

Requires: pip install "mcp[cli]"
"""
from mcp.server.fastmcp import FastMCP

from src.tools.market_data import get_price, get_fundamentals, get_news
from src.graph import run as run_analysis

mcp = FastMCP("equitycrew")


@mcp.tool()
def price(ticker: str) -> dict:
    """Get the latest price and day change for a stock ticker."""
    return get_price(ticker)


@mcp.tool()
def fundamentals(ticker: str) -> dict:
    """Get key fundamentals (market cap, P/E, margins, growth) for a ticker."""
    return get_fundamentals(ticker)


@mcp.tool()
def news(ticker: str) -> dict:
    """Get recent headlines for a ticker."""
    return get_news(ticker)


@mcp.tool()
def full_analysis(ticker: str) -> str:
    """Run the full multi-agent research crew and return an investment memo."""
    return run_analysis(ticker)


if __name__ == "__main__":
    mcp.run()

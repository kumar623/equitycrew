"""Real-time market-data tools. These are the ONLY source of numbers —
agents must never invent figures, they call these.

Uses yfinance (no API key needed). Finnhub is optional for richer news.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

import yfinance as yf


def _safe(d: dict, key: str, default: Any = None):
    v = d.get(key, default)
    return v if v is not None else default


def get_price(ticker: str) -> dict:
    """Live-ish quote: current/last price, day change, currency."""
    t = yf.Ticker(ticker)
    info = t.fast_info if hasattr(t, "fast_info") else {}
    try:
        last = float(info["last_price"])
        prev = float(info["previous_close"])
        change_pct = round((last - prev) / prev * 100, 2) if prev else None
        return {
            "ticker": ticker.upper(),
            "price": round(last, 2),
            "previous_close": round(prev, 2),
            "change_pct": change_pct,
            "currency": _safe(dict(info), "currency", "USD"),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker.upper(), "error": f"price unavailable: {e}"}


def get_fundamentals(ticker: str) -> dict:
    """Key fundamentals: market cap, P/E, margins, revenue growth."""
    t = yf.Ticker(ticker)
    try:
        info = t.info  # richer but slower dict
        return {
            "ticker": ticker.upper(),
            "name": _safe(info, "shortName"),
            "sector": _safe(info, "sector"),
            "market_cap": _safe(info, "marketCap"),
            "trailing_pe": _safe(info, "trailingPE"),
            "forward_pe": _safe(info, "forwardPE"),
            "profit_margin": _safe(info, "profitMargins"),
            "revenue_growth": _safe(info, "revenueGrowth"),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker.upper(), "error": f"fundamentals unavailable: {e}"}


def get_news(ticker: str, limit: int = 5) -> dict:
    """Recent headlines for the ticker."""
    t = yf.Ticker(ticker)
    try:
        raw = getattr(t, "news", []) or []
        items = []
        for n in raw[:limit]:
            content = n.get("content", n)  # yfinance shapes vary by version
            items.append(
                {
                    "title": content.get("title") or n.get("title"),
                    "publisher": (content.get("provider") or {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else n.get("publisher"),
                }
            )
        return {"ticker": ticker.upper(), "headlines": items,
                "as_of": datetime.now(timezone.utc).isoformat()}
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker.upper(), "error": f"news unavailable: {e}"}


def _quotes(ticker: str) -> bool:
    """True when the symbol actually returns a tradable price."""
    return get_price(ticker).get("price") is not None


def resolve_ticker(ticker: str) -> dict:
    """Validate a symbol before spending a full crew run on it.

    Returns {"ok": True} when it quotes. Otherwise returns suggestions that
    were themselves verified to quote — most often the exchange-suffixed form
    Indian listings need (RELIANCE -> RELIANCE.NS), since Yahoo only serves
    plain symbols for US listings.
    """
    sym = ticker.strip().upper()
    if not sym:
        return {"ok": False, "suggestions": [], "reason": "No ticker entered."}
    if _quotes(sym):
        return {"ok": True, "ticker": sym}

    suggestions = []
    candidates = []
    if "." not in sym:
        candidates += [f"{sym}.NS", f"{sym}.BO"]   # NSE, then BSE
    # a few symbols people reach for that no longer trade under that name
    candidates += {"HDFC": ["HDFCBANK.NS", "HDB"]}.get(sym, [])
    for c in candidates:
        if len(suggestions) >= 3:
            break
        if _quotes(c):
            suggestions.append(c)
    return {
        "ok": False,
        "ticker": sym,
        "suggestions": suggestions,
        "reason": f"“{sym}” returned no market data — it may be delisted, "
                  f"renamed, or listed on a non-US exchange.",
    }


if __name__ == "__main__":
    import json, sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(json.dumps(get_price(sym), indent=2))
    print(json.dumps(get_fundamentals(sym), indent=2))
    print(json.dumps(get_news(sym), indent=2))

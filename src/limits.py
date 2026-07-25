"""Spend protection for the public demo.

The research endpoint costs real money per call (~$0.10), so a public URL with
no limits is an open tap on the owner's API credit. Three cheap defences, in
order of how often they save money:

  1. Result cache — repeat requests for the same ticker replay a stored run.
  2. Per-IP hourly limit — stops one visitor looping the endpoint.
  3. Global daily cap — bounds the worst case no matter how many visitors.

State is in-process: this runs as a single container, so a dict is enough and
avoids a Redis dependency for a portfolio demo. Restarting clears it.
"""
from __future__ import annotations
import os
import threading
import time
from collections import defaultdict, deque
from datetime import date

CACHE_TTL_S = int(os.getenv("EQUITYCREW_CACHE_TTL", "900"))       # 15 min
IP_LIMIT_PER_HOUR = int(os.getenv("EQUITYCREW_IP_LIMIT", "5"))
DAILY_RUN_LIMIT = int(os.getenv("EQUITYCREW_DAILY_LIMIT", "60"))  # ~$6/day ceiling

_lock = threading.Lock()
_cache: dict = {}                      # ticker -> (stored_at, [payloads])
_ip_hits: dict = defaultdict(deque)    # ip -> timestamps of billed runs
_day = {"date": date.today(), "count": 0}


def cached(ticker: str):
    """Stored payload list for a recent run of this ticker, or None."""
    with _lock:
        entry = _cache.get(ticker.upper())
        if not entry:
            return None
        stored_at, payloads = entry
        if time.time() - stored_at > CACHE_TTL_S:
            _cache.pop(ticker.upper(), None)
            return None
        return payloads


def store(ticker: str, payloads: list) -> None:
    with _lock:
        _cache[ticker.upper()] = (time.time(), payloads)


def any_cached():
    """Most recent cached run, used to show something useful once capped."""
    with _lock:
        fresh = [(ts, t, p) for t, (ts, p) in _cache.items()
                 if time.time() - ts <= CACHE_TTL_S]
    if not fresh:
        return None, None
    ts, ticker, payloads = max(fresh)
    return ticker, payloads


def check(ip: str) -> dict:
    """Decide whether a *billed* run may proceed. Cache hits never call this."""
    now = time.time()
    with _lock:
        if _day["date"] != date.today():          # new day, reset the counter
            _day["date"], _day["count"] = date.today(), 0
        if _day["count"] >= DAILY_RUN_LIMIT:
            return {"ok": False, "scope": "daily",
                    "message": f"This demo runs at most {DAILY_RUN_LIMIT} fresh "
                               f"analyses a day to bound API costs. Showing the "
                               f"most recent cached run instead."}
        hits = _ip_hits[ip]
        while hits and now - hits[0] > 3600:       # slide the 1-hour window
            hits.popleft()
        if len(hits) >= IP_LIMIT_PER_HOUR:
            wait_min = max(1, int((3600 - (now - hits[0])) // 60))
            return {"ok": False, "scope": "ip",
                    "message": f"You've run {IP_LIMIT_PER_HOUR} analyses this "
                               f"hour — the limit for this demo. Try again in "
                               f"~{wait_min} min, or pick a ticker someone ran "
                               f"recently to see a cached result instantly."}
        return {"ok": True}


def record(ip: str) -> None:
    """Count a run that actually spent money."""
    with _lock:
        _ip_hits[ip].append(time.time())
        _day["count"] += 1


def stats() -> dict:
    with _lock:
        return {"runs_today": _day["count"], "daily_limit": DAILY_RUN_LIMIT,
                "ip_limit_per_hour": IP_LIMIT_PER_HOUR,
                "cached_tickers": sorted(_cache.keys()),
                "cache_ttl_s": CACHE_TTL_S}

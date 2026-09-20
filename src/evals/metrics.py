"""Eval-only helpers: sabotage builders for the catch-rate tests.

The deterministic numeric checks used to live here. They now live in
src/checks.py because the verifier runs them in production — this module
re-exports them so eval code keeps importing from one place.
"""
from __future__ import annotations
import re

from ..checks import (  # noqa: F401  (re-exported for eval callers)
    TOKEN, extract_numbers, numeric_accuracy, truth_values,
)


# ---------- sabotage builders (for catch-rate tests) ----------

def perturb_numbers(memo: str, count: int = 4):
    """Corrupt up to `count` clearly-financial numbers (~+17% each).
    Returns (sabotaged_memo, planted) where planted = [(original, corrupted)]."""
    planted = []
    sabotaged = memo
    seen = set()
    for m in TOKEN.finditer(memo):
        if len(planted) >= count:
            break
        if not (m.group("prefix") or m.group("suffix")):
            continue
        original = m.group(0).strip()
        if original in seen or original not in sabotaged:
            continue
        seen.add(original)
        raw = m.group("num")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        bad_num = f"{float(raw.replace(',', '')) * 1.17:.{decimals}f}"
        corrupted = original.replace(raw, bad_num)
        sabotaged = sabotaged.replace(original, corrupted, 1)
        planted.append((original, corrupted))
    return sabotaged, planted


def critic_sabotage_variants(memo: str) -> dict:
    """Structurally-broken memos the critic is expected to reject.
    Variants that come out empty or unchanged are dropped (not a valid test)."""
    variants = {
        "missing_rating": re.sub(r"(?im)^.*\brating\b.*$", "", memo),
        "unsupported_claim": memo + (
            "\n\nManagement has privately guided to 250% revenue growth next "
            "quarter and the CFO has personally guaranteed the stock will double."
        ),
        "truncated": memo[: len(memo) // 3],
    }
    return {
        name: v for name, v in variants.items()
        if v.strip() and v.strip() != memo.strip()
    }

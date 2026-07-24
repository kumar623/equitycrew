"""Deterministic metrics: extract numeric claims from a memo and check them
against tool data. No LLM involved — these checks must be trustworthy."""
from __future__ import annotations
import re

# $210.11 | 62.97% | 32.17x | $5.09T | 85.2 — number + optional money/percent/scale marker
TOKEN = re.compile(
    r"(?P<prefix>\$)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suffix>%|x\b|T\b|B\b|M\b|trillion|billion)?",
    re.I,
)


def truth_values(*data_dicts: dict) -> set:
    """Every numeric value from tool data, plus the variants a memo might use
    (0.6297 -> 62.97%, 5_090_000_000_000 -> 5.09T / 5090B)."""
    vals = set()
    for d in data_dicts:
        for v in (d or {}).values():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            vals.add(float(v))
            if 0 < abs(v) < 1:
                vals.add(round(v * 100, 4))          # ratio -> percent
            if abs(v) > 1e9:
                vals.update((v / 1e12, v / 1e9, v / 1e6))  # T / B / M scales
    return vals


def extract_numbers(memo: str) -> list:
    """Financial-looking numbers in the memo: [(text, value, decimals), ...].
    Plain small integers and bare years are skipped (they're rarely data claims)."""
    out = []
    for m in TOKEN.finditer(memo):
        raw = m.group("num")
        has_marker = bool(m.group("prefix") or m.group("suffix"))
        has_decimals = "." in raw
        value = float(raw.replace(",", ""))
        if not has_marker and not has_decimals:
            continue
        if not has_marker and 1900 <= value <= 2100:
            continue
        decimals = len(raw.split(".")[1]) if has_decimals else 0
        out.append((m.group(0).strip(), value, decimals))
    return out


def _matches(value: float, decimals: int, truth: float) -> bool:
    if abs(truth) > 1e-9 and abs(value - truth) / abs(truth) < 0.005:
        return True
    return abs(round(truth, decimals) - value) < 10 ** -(decimals + 4)


def numeric_accuracy(memo: str, *data_dicts: dict):
    """Fraction of the memo's financial numbers that match a tool-data value
    (exactly, or as a rounding of it). Returns (accuracy, ungrounded_tokens)."""
    truths = truth_values(*data_dicts)
    numbers = extract_numbers(memo)
    if not numbers:
        return 1.0, []
    ungrounded = [
        text for text, value, decimals in numbers
        if not any(_matches(value, decimals, t) for t in truths)
    ]
    return 1 - len(ungrounded) / len(numbers), ungrounded


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

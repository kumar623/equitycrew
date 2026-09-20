"""Deterministic numeric checks — no LLM involved.

These are the trustworthy half of verification. The verifier agent asks Claude
whether the memo's figures match fresh tool data; this module answers the same
question in code, so the two verdicts can be compared instead of the LLM being
taken at its word.

It lived in src/evals/ for most of this project's life, grading the pipeline
from the outside. It is production code now, for the same reason section
completeness moved out of the critic's prompt: whether a number matches is a
fact, not a judgement.

Scope, stated honestly: a figure counts as grounded when it matches a value the
tools returned, or a rounding/rescaling of one. A *derived* figure the writer
computed legitimately ("margins expanded 12% YoY") has no counterpart in the
tool data and so reads as ungrounded here. Ungrounded therefore means "not
directly traceable to tool data", NOT "wrong" — which is exactly why this
supplements the LLM check rather than replacing it.
"""
from __future__ import annotations
import re

# $210.11 | 62.97% | 32.17x | $5.09T | 85.2 — number + optional money/percent/scale marker
TOKEN = re.compile(
    r"(?P<prefix>\$)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suffix>%|x\b|T\b|B\b|M\b|trillion|billion)?",
    re.I,
)


#: Tool fields stored as a fraction but written as a percentage in a memo
#: (revenue_growth 1.059 -> "105.9%"). Matched on the field NAME, because
#: magnitude is not a reliable signal: the rule here used to be "convert
#: anything below 1", which silently excluded every growth rate above 100%.
#: NVDA grew 105.9%, the memo said so correctly, and the checker called it
#: ungrounded three times. Fields already in percent (change_pct) must not be
#: converted, so they are matched by neither rule.
FRACTION_FIELDS = re.compile(r"margin|growth|yield|payout", re.I)


def truth_values(*data_dicts: dict) -> set:
    """Every numeric value from tool data, plus the variants a memo might use
    (0.6297 -> 62.97%, 1.059 -> 105.9%, 5_090_000_000_000 -> 5.09T / 5090B)."""
    vals = set()
    for d in data_dicts:
        for key, v in (d or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            vals.add(float(v))
            # ratio -> percent, by field name or (for unlabelled dicts) by size
            if FRACTION_FIELDS.search(key) or 0 < abs(v) < 1:
                vals.add(round(v * 100, 4))
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

"""Pure validation functions — the deterministic safety floor of the agent.
No API calls, no LLM, just logic.

Note: function-calling enforces JSON Schema types, so we don't defensively
isinstance-check every input. We trust the schema and validate the *content*.
"""
from __future__ import annotations
import re
from datetime import date


def is_real_date(dob: str) -> bool:
    """Strict YYYY-MM-DD check that the date actually exists.
    False for 1989-02-29 (not a leap year), 2024-13-01, 'foo', etc."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", dob or ""):
        return False
    try:
        y, m, d = dob.split("-")
        date(int(y), int(m), int(d))
        return True
    except ValueError:
        return False


def is_aadhaar_last4(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", s or ""))


def is_pincode(s: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", s or ""))


def luhn_ok(card_number: str) -> bool:
    digits = [int(c) for c in card_number if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def expiry_ok(month: int, year: int) -> bool:
    """Real month, not in the past. We don't bound the future — let the API decide."""
    if not (1 <= month <= 12):
        return False
    today = date.today()
    return (year, month) >= (today.year, today.month)
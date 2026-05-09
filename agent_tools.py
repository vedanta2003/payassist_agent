"""LLM-facing tools — the only functions the LLM can invoke.

Invariants enforced here (not in the prompt):
  • Sequencing: each tool refuses if its prerequisites haven't run.
  • Retry limits: counted server-side; the LLM cannot extend them.
  • Sensitive data: dob/aadhaar/pincode never leave the Session.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

import api_client
from api_client import APIError
from validators import is_real_date, is_aadhaar_last4, is_pincode, luhn_ok, expiry_ok

log = logging.getLogger("payassist")

MAX_VERIFY_ATTEMPTS  = 3
MAX_PAYMENT_ATTEMPTS = 3


@dataclass
class Session:
    account_id:     Optional[str]   = None
    full_name:      Optional[str]   = None     # never returned to LLM
    balance:        Optional[float] = None     # never returned to LLM until verified
    _dob:           Optional[str]   = None
    _aadhaar_last4: Optional[str]   = None
    _pincode:       Optional[str]   = None

    verified:         bool = False
    verify_attempts:  int  = 0
    verify_locked:    bool = False
    payment_attempts: int  = 0
    payment_locked:   bool = False


# ── Tool implementations ──────────────────────────────────────────────

def lookup_account(session: Session, account_id: str) -> dict:
    """Confirm an account exists. Returns nothing else — name and balance
    stay server-side until verification succeeds."""
    aid = (account_id or "").strip().upper()
    if not re.fullmatch(r"ACC\d{3,8}", aid):
        return {"ok": False, "error": "invalid_format",
                "message": "Account ID must look like ACC followed by 3-8 digits."}
    try:
        data = api_client.lookup_account(aid)
    except APIError as e:
        return {"ok": False, "error": e.code, "message": e.message}

    session.account_id     = data["account_id"]
    session.full_name      = data["full_name"]
    session.balance        = float(data.get("balance", 0))
    session._dob           = data.get("dob")
    session._aadhaar_last4 = data.get("aadhaar_last4")
    session._pincode       = data.get("pincode")
    # Reset downstream state in case lookup is re-run mid-flow
    session.verified = False
    session.verify_attempts = 0
    session.verify_locked = False
    return {"ok": True, "account_exists": True}


def verify_identity(
    session: Session,
    full_name: str,
    dob: Optional[str] = None,
    aadhaar_last4: Optional[str] = None,
    pincode: Optional[str] = None,
) -> dict:
    """Strict verification. Pass: name matches exactly AND one factor matches.
    Burns an attempt on a real mismatch; format errors do not."""
    if session.account_id is None:
        return {"ok": False, "error": "no_account",
                "message": "Look up an account before verifying."}
    if session.verify_locked:
        return {"ok": False, "error": "locked",
                "message": "Verification is locked for this session."}
    if not full_name or len(full_name.strip()) < 2:
        return {"ok": False, "error": "missing_name",
                "message": "A full name is required."}
    if not (dob or aadhaar_last4 or pincode):
        return {"ok": False, "error": "missing_secondary",
                "message": "Need one of dob, aadhaar_last4, or pincode. "
                           "If a previous call returned secondary_match: true, "
                           "re-supply that same factor along with the corrected name."}

    # Format pre-checks — these do NOT burn an attempt
    if dob is not None and not is_real_date(dob):
        return {"ok": False, "error": "invalid_dob_format",
                "message": "DOB must be a real calendar date in YYYY-MM-DD format. "
                           "Tip: 29 Feb only exists in leap years.",
                "burned_attempt": False}
    if aadhaar_last4 is not None and not is_aadhaar_last4(aadhaar_last4):
        return {"ok": False, "error": "invalid_aadhaar_format",
                "message": "Aadhaar last 4 must be exactly 4 digits.",
                "burned_attempt": False}
    if pincode is not None and not is_pincode(pincode):
        return {"ok": False, "error": "invalid_pincode_format",
                "message": "Pincode must be exactly 6 digits.",
                "burned_attempt": False}

    name_match = (full_name == session.full_name)
    if dob is not None:
        factor_used, secondary_match = "dob", (dob == session._dob)
    elif aadhaar_last4 is not None:
        factor_used, secondary_match = "aadhaar_last4", (aadhaar_last4 == session._aadhaar_last4)
    else:
        factor_used, secondary_match = "pincode", (pincode == session._pincode)

    if name_match and secondary_match:
        session.verified = True
        log.info(f"[VERIFY] ✓ verified ({factor_used})")
        return {"ok": True, "verified": True, "balance": session.balance}

    session.verify_attempts += 1
    remaining = MAX_VERIFY_ATTEMPTS - session.verify_attempts
    if remaining <= 0:
        session.verify_locked = True
        log.info(f"[VERIFY] ✗ locked after {session.verify_attempts} attempts")
        return {"ok": False, "verified": False, "locked": True, "attempts_remaining": 0,
                "message": "Verification locked after too many failed attempts."}
    return {"ok": False, "verified": False,
            "name_match": name_match, "secondary_match": secondary_match,
            "secondary_factor_attempted": factor_used,
            "attempts_remaining": remaining,
            "message": "Identity details did not match."}


def validate_payment_amount(session: Session, amount: float) -> dict:
    """Check that an amount is well-formed and within balance."""
    if not session.verified:
        return {"ok": False, "error": "not_verified"}
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_amount", "message": "Amount must be a number."}
    if amount <= 0:
        return {"ok": False, "error": "invalid_amount", "message": "Amount must be greater than zero."}
    if round(amount, 2) != amount:
        return {"ok": False, "error": "invalid_amount", "message": "At most 2 decimal places."}
    if amount > session.balance:
        return {"ok": False, "error": "invalid_amount",
                "message": f"Amount exceeds balance of ₹{session.balance:.2f}."}
    return {"ok": True, "amount": amount}


def process_payment(
    session: Session,
    amount: float,
    card_number: str,
    cvv: str,
    expiry_month: int,
    expiry_year: int,
    cardholder_name: str,
) -> dict:
    """Validate card fields locally, then charge via the API.
    Combines what used to be validate_card_details + process_payment —
    one round-trip, errors clearly attributed."""
    if not session.verified:
        return {"ok": False, "error": "not_verified"}
    if session.payment_locked:
        return {"ok": False, "error": "payment_locked",
                "message": "Payment is locked after too many failures."}

    # Re-check amount in case the LLM passes a different one
    amt_check = validate_payment_amount(session, amount)
    if not amt_check["ok"]:
        return amt_check
    amount = amt_check["amount"]

    # Local card validation
    clean_num = re.sub(r"[\s-]", "", card_number or "")
    if not clean_num.isdigit() or not luhn_ok(clean_num):
        return {"ok": False, "error": "invalid_card",
                "message": "Card number is invalid (failed Luhn check)."}
    expected_cvv = 4 if clean_num.startswith(("34", "37")) else 3
    cvv_clean = (cvv or "").strip()
    if not cvv_clean.isdigit() or len(cvv_clean) != expected_cvv:
        return {"ok": False, "error": "invalid_cvv",
                "message": f"CVV must be {expected_cvv} digits for this card."}
    try:
        m, y = int(expiry_month), int(expiry_year)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_expiry",
                "message": "Expiry month and year must be integers."}
    if y < 100:
        y += 2000
    if not expiry_ok(m, y):
        return {"ok": False, "error": "invalid_expiry",
                "message": "Card expiry is invalid or in the past."}
    if not cardholder_name or len(cardholder_name.strip()) < 2:
        return {"ok": False, "error": "invalid_cardholder",
                "message": "Cardholder name is required."}

    # Submit to API
    try:
        result = api_client.process_payment(
            account_id=session.account_id, amount=amount,
            card_number=clean_num, cvv=cvv_clean,
            expiry_month=m, expiry_year=y,
            cardholder_name=cardholder_name.strip(),
        )
        return {"ok": True,
                "transaction_id": result.get("transaction_id"),
                "amount": amount,
                "card_last4": clean_num[-4:]}
    except APIError as e:
        retryable = e.code in {"invalid_card", "invalid_cvv", "invalid_expiry"}
        session.payment_attempts += 1
        if retryable and session.payment_attempts < MAX_PAYMENT_ATTEMPTS:
            return {"ok": False, "error": e.code, "retryable": True,
                    "attempts_remaining": MAX_PAYMENT_ATTEMPTS - session.payment_attempts,
                    "message": e.message}
        if retryable:
            session.payment_locked = True
        return {"ok": False, "error": e.code, "retryable": False, "message": e.message}


# ── Tool schemas (sent to OpenAI) ─────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_account",
            "description": (
                "Look up an account by ID. Confirms whether it exists. "
                "Does NOT return name, balance, or verification details — those "
                "are released only after verify_identity succeeds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Format ACC + 3-8 digits, e.g. ACC1001."},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_identity",
            "description": (
                "Strictly verify the user. Requires full name AND one secondary factor "
                "(dob in YYYY-MM-DD, aadhaar_last4 as 4 digits, or pincode as 6 digits). "
                "3 attempts max; format errors don't burn an attempt. "
                "On retries: ALWAYS re-supply any factor that previously matched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name":     {"type": "string", "description": "Exact casing as stated by user."},
                    "dob":           {"type": "string", "description": "YYYY-MM-DD."},
                    "aadhaar_last4": {"type": "string", "description": "4 digits."},
                    "pincode":       {"type": "string", "description": "6 digits."},
                },
                "required": ["full_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_payment_amount",
            "description": (
                "Check that an amount is well-formed and within balance. "
                "Optional — process_payment re-validates anyway. Useful for "
                "confirming with the user before collecting card details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "INR amount, e.g. 500.00."},
                },
                "required": ["amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_payment",
            "description": (
                "Validate card fields locally and submit the payment. "
                "Call ONLY after the user has explicitly confirmed (e.g. said 'yes' "
                "to a confirmation prompt). This actually charges the card."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount":          {"type": "number"},
                    "card_number":     {"type": "string", "description": "Digits only."},
                    "cvv":             {"type": "string", "description": "3 or 4 digits."},
                    "expiry_month":    {"type": "integer", "description": "1-12."},
                    "expiry_year":     {"type": "integer", "description": "4-digit year."},
                    "cardholder_name": {"type": "string"},
                },
                "required": ["amount", "card_number", "cvv", "expiry_month", "expiry_year", "cardholder_name"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_account":          lookup_account,
    "verify_identity":         verify_identity,
    "validate_payment_amount": validate_payment_amount,
    "process_payment":         process_payment,
}
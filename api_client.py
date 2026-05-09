"""HTTP calls to Prodigal's payment API.
This is NOT what the LLM calls directly — see agent_tools.py for that.
"""
from __future__ import annotations
import logging
import requests

log = logging.getLogger("payassist")

BASE_URL = "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com/api"
TIMEOUT = 15


class APIError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def lookup_account(account_id: str) -> dict:
    """Returns full account dict on success. Raises APIError on failure.
    The caller is responsible for stripping sensitive fields before
    exposing this to the LLM."""
    try:
        log.info(f"[API] lookup_account → account_id: {account_id}")
        resp = requests.post(
            f"{BASE_URL}/lookup-account",
            json={"account_id": account_id},
            timeout=TIMEOUT,
        )
        data = resp.json()
        if resp.status_code == 200:
            log.info(f"[API] lookup_account ✓ → name: {data.get('full_name')}, balance: ₹{data.get('balance')}")
            return data
        code = data.get("error_code", "unknown_error")
        msg = data.get("message", "Account lookup failed.")
        log.info(f"[API] lookup_account ✗ → {code}: {msg}")
        raise APIError(code, msg)
    except requests.exceptions.Timeout:
        raise APIError("timeout", "The server took too long to respond. Please try again.")
    except requests.exceptions.ConnectionError:
        raise APIError("connection_error", "Could not connect to the server. Please check your internet.")
    except APIError:
        raise
    except Exception as e:
        raise APIError("unexpected_error", f"Something went wrong: {e}")


def process_payment(
    account_id: str,
    amount: float,
    card_number: str,
    cvv: str,
    expiry_month: int,
    expiry_year: int,
    cardholder_name: str,
) -> dict:
    """Returns {success, transaction_id} on success. Raises APIError on failure."""
    payload = {
        "account_id": account_id,
        "amount": round(float(amount), 2),
        "payment_method": {
            "type": "card",
            "card": {
                "cardholder_name": cardholder_name,
                "card_number": card_number,
                "cvv": cvv,
                "expiry_month": int(expiry_month),
                "expiry_year": int(expiry_year),
            },
        },
    }
    try:
        log.info(f"[API] process_payment → account: {account_id}, amount: ₹{amount}")
        resp = requests.post(f"{BASE_URL}/process-payment", json=payload, timeout=TIMEOUT)
        data = resp.json()
        if resp.status_code == 200 and data.get("success"):
            log.info(f"[API] process_payment ✓ → txn_id: {data.get('transaction_id')}")
            return data
        code = data.get("error_code", "payment_failed")
        messages = {
            "insufficient_balance": "The amount exceeds the outstanding balance.",
            "invalid_card":         "The card number was rejected by the payment processor.",
            "invalid_cvv":          "The CVV was rejected by the payment processor.",
            "invalid_expiry":       "The card expiry was rejected (likely expired).",
            "invalid_amount":       "The amount is invalid.",
        }
        msg = messages.get(code, data.get("message", "Payment failed."))
        log.info(f"[API] process_payment ✗ → {code}: {msg}")
        raise APIError(code, msg)
    except requests.exceptions.Timeout:
        raise APIError("timeout", "Payment request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise APIError("connection_error", "Could not connect to the payment server.")
    except APIError:
        raise
    except Exception as e:
        raise APIError("unexpected_error", f"Something went wrong: {e}")

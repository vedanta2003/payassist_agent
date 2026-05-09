"""Agent — function-calling loop over OpenAI's Chat Completions API.

One Agent per session. Each call to next(user_input):
  1. Append user message to history.
  2. Call the model. If it requests tools, run them, append results, loop.
  3. Once the model emits text, return it.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Optional

from openai import OpenAI
from agent_tools import Session, TOOL_SCHEMAS, TOOL_FUNCTIONS
from prompts import SYSTEM_PROMPT

log = logging.getLogger("payassist")

# Hardcoded — gpt-4o is required for this agent. gpt-4o-mini was evaluated in the
# eval suite and failed the payment-confirmation hard rule on 4/22 adversarial
# scenarios (it skipped the explicit "yes" step before charging). For a payment
# agent, that's a non-negotiable failure mode, so we lock the model here.
MODEL         = "gpt-4o"
MAX_TOOL_HOPS = 8       # safety cap on tool calls per turn
TEMPERATURE   = 0


_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI()
    return _client


class Agent:
    """Required interface: next(user_input) -> {'message': str}."""

    def __init__(self):
        self._session = Session()
        self._messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Per-turn tool-call log, for evals/debugging. Each entry:
        # {"turn": int, "name": str, "args": dict (with card data masked), "result": dict}
        self.tool_calls: list[dict] = []
        self._turn = 0

    def next(self, user_input: str) -> dict:
        self._turn += 1
        text = (user_input or "").strip() or "(user just opened the chat)"
        self._messages.append({"role": "user", "content": text})
        try:
            return {"message": self._run_loop()}
        except Exception:
            log.exception("[AGENT] error in run loop")
            msg = ("I'm having trouble right now. Could you rephrase, "
                   "or contact support if the issue persists?")
            self._messages.append({"role": "assistant", "content": msg})
            return {"message": msg}

    def _run_loop(self) -> str:
        for _ in range(MAX_TOOL_HOPS):
            resp = _get_client().chat.completions.create(
                model=MODEL, messages=self._messages,
                tools=TOOL_SCHEMAS, tool_choice="auto",
                temperature=TEMPERATURE,
            )
            msg = resp.choices[0].message
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self._messages.append(entry)

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                # Mask card data in logs
                log_args = dict(args)
                if "card_number" in log_args:
                    n = str(log_args["card_number"])
                    log_args["card_number"] = "*" * max(0, len(n) - 4) + n[-4:]
                if "cvv" in log_args:
                    log_args["cvv"] = "***"
                log.info(f"[TOOL] {name}({log_args})")

                fn = TOOL_FUNCTIONS.get(name)
                if fn is None:
                    result = {"ok": False, "error": "unknown_tool"}
                else:
                    try:
                        result = fn(self._session, **args)
                    except TypeError as e:
                        result = {"ok": False, "error": "bad_arguments", "message": str(e)}
                    except Exception:
                        log.exception(f"[TOOL] {name} crashed")
                        result = {"ok": False, "error": "tool_crash"}

                log.info(f"[TOOL] {name} → {json.dumps(result)[:300]}")
                # Record for eval — masked args, full result (already redacted by tool design)
                self.tool_calls.append({
                    "turn":   self._turn,
                    "name":   name,
                    "args":   log_args,
                    "result": result,
                })
                self._messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        log.warning("[AGENT] hit MAX_TOOL_HOPS")
        return ("I'm having trouble completing that step. "
                "Could you rephrase, or contact support?")
# PayAssist

LLM-driven payment collection agent. Conversational, end-to-end: greets the user, looks up an account, verifies identity strictly, collects card details, charges, and recaps — all over chat. Single-file `Agent` class implementing the required `next(user_input) -> {"message": str}` interface.

Built for the Prodigal Agent Engineer take-home.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/fastapi-0.110+-009688)
![Model](https://img.shields.io/badge/model-gpt--4o-412991)

---

## Quick start

### 1. Clone the repo

```bash
git clone https://github.com/vedanta2003/payassist_agent.git
cd payassist_agent
```

### 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 5. Pick how you want to run it

#### I. Run in Web App (recommended)

```bash
uvicorn server:app --reload --port 8000
```

### OR

#### II. Run in terminal

```bash
python cli.py
```

The web UI shows the chat on the left and a **live stream of every tool call, API call, and verification event** on the right — useful for understanding what the agent is doing internally.

---

## Architecture

```
agent.py        Function-calling loop. One Agent per session.
prompts.py      System prompt — encodes the flow, hard rules, and worked examples.
agent_tools.py  Five LLM-facing tools. Session state lives here.
api_client.py   HTTP client for the Prodigal payment API.
validators.py   Pure deterministic checks (Luhn, leap year, etc).
server.py       FastAPI HTTP wrapper + Server-Sent Events for log streaming.
static/         Single-page web UI (vanilla HTML/JS, no build step).
cli.py          Interactive REPL for terminal testing.
eval.py         22 formal eval scenarios, ~65 assertions, strict pass/fail.
```

### How a turn works

1. User message hits `Agent.next(user_input)`.
2. Message appended to history; OpenAI Chat Completions called with the system prompt + tool schemas.
3. If the model requests tools, each runs locally against the session, the result is appended to history, and the loop continues.
4. Once the model emits plain text, that becomes the response.

### What's a tool, what's a prompt

The split is deliberate.

**Tools own** (deterministic safety floor):
- Strict comparisons: name equality (case-sensitive), Luhn check, leap-year date validity
- Sequencing enforcement: payment refuses if `not session.verified`, verify refuses if no lookup, etc.
- Retry counters (3 verification attempts, 3 payment attempts) — server-side, not visible to the LLM
- Sensitive data isolation: DOB, Aadhaar, and pincode live in `Session._dob`, `_aadhaar_last4`, `_pincode` — only `verify_identity` can read them; they never appear in the LLM's context

**Prompt owns** (the "brain"):
- Free-form text → structured fields ("14th May 1990" → `1990-05-14`)
- Flow control: what to ask next given conversation state
- Tone: professional, concise, doesn't over-apologize, holds firm under pressure
- Refusal patterns: PII fishing, authority claims, jailbreak roleplay, sequence skipping

This split is intentional. We tried letting tools track flow state (caching matched verification factors so retries auto-carry-forward) and rolled it back — that's prompt territory, not tool territory. We tried letting the prompt do its own date validation ("29 Feb only exists in leap years") and that caused two regressions where the model hallucinated leap-year status. The rule we converged on: **tools answer questions of fact, prompts make decisions.**

### Why function calling rather than LangGraph

LangGraph is the right choice for graphs with cycles, parallel branches, or multi-agent supervision. Our flow is linear: greet → lookup → verify → amount → card → confirm → pay. Function calling expresses the same logic with less code and one fewer dependency. Tools enforce sequencing server-side, which gives the same "can't skip steps" guarantee a graph would. See the design doc for the full trade-off.

### Why gpt-4o, not mini

We evaluated both. `gpt-4o-mini` consistently failed the explicit-payment-confirmation hard rule — it would skip the "yes" step and charge on receiving card details. For a payment agent that's a non-negotiable failure mode, so the model is hardcoded to `gpt-4o` in `agent.py`.

---

## Test accounts

| Account ID | Name                            | DOB        | Aadhaar | Pincode | Balance   |
|------------|---------------------------------|------------|---------|---------|-----------|
| ACC1001    | Nithin Jain                     | 1990-05-14 | 4321    | 400001  | ₹1,250.75 |
| ACC1002    | Rajarajeswari Balasubramaniam   | 1985-11-23 | 9876    | 400002  | ₹540.00   |
| ACC1003    | Priya Agarwal                   | 1992-08-10 | 2468    | 400003  | ₹0.00     |
| ACC1004    | Rahul Mehta                     | 1988-02-29 | 1357    | 400004  | ₹3,200.50 |

Successful payment card: `4532015112830366`, CVV `123`, expiry `12/2027`.

Note ACC1004's DOB — leap-year edge case. The agent passes `1988-02-29` to `verify_identity` and lets the tool decide. `1989-02-29` is rejected by the tool with `invalid_dob_format` and does not burn a retry attempt.

---

## Evaluation

`eval.py` runs 22 scenarios with ~65 strict pass/fail assertions of four kinds:

- **Tool-call introspection**: was X tool called? With what args? At most N times?
- **State assertions**: `session.verified`, `verify_locked`, payment processed/not
- **Content assertions**: `no_message_contains` (catches PII leaks), `payment_processed_after_explicit_confirmation` (catches stale-yes bugs)
- **LLM-judge** (uses `gpt-4o-mini` as evaluator): for fuzzier checks like "did the agent acknowledge zero balance gracefully?"

Strict pass: every assertion in a scenario must pass for the scenario to pass.

```bash
python eval.py                       # full run, ~5-8 min
python eval.py --quick               # skip LLM judges, ~3 min
python eval.py --filter pii          # one category
python eval.py --out my_baseline.json
```

Exit code is 0 on full pass, 1 otherwise (CI-friendly). See `baseline_4o_full.json` for the canonical pass rate at submission time.

---

## Sample conversations

### Successful end-to-end payment

```
User:  Hi, I'm Nithin Jain, my account is ACC1001
Agent: Thank you. To verify your identity, could you please provide your date
       of birth, the last 4 digits of your Aadhaar, or your pincode?

User:  DOB 1990-05-14
Agent: Verified, Nithin. Your outstanding balance is ₹1,250.75. How much
       would you like to pay today?

User:  500
Agent: Great — ₹500 is valid. Please share your card number, CVV, expiry
       (month and year), and cardholder name.

User:  4532015112830366, CVV 123, exp 12/2027, name Nithin Jain
Agent: To confirm, paying ₹500 using the card ending in 0366. Shall I proceed?

User:  yes
Agent: Payment of ₹500 processed successfully. Transaction ID:
       txn_1778346646384_yufq1ir.
```

### Verification failure with lockout

```
User:  ACC1001
Agent: To verify your identity, please share your full name and either
       your date of birth, last 4 of Aadhaar, or pincode.

User:  John Doe, DOB 2000-01-01
Agent: The details you provided didn't match our records. You have 2
       attempts remaining...

User:  Jane Doe, DOB 2000-01-02
Agent: Still no match. You have 1 attempt remaining...

User:  Bob Smith, DOB 2000-01-03
Agent: Your account has been locked due to too many failed verification
       attempts. For further assistance, please contact our support team
       at support@payassist.example.com or call 1800-PAY-ASSIST.
```

### Payment failure with retry (Luhn)

```
[verified, amount 200 confirmed]

User:  card 1234567890123456, CVV 123, exp 12/2027, name Nithin Jain
Agent: To confirm, paying ₹200 using the card ending in 3456. Shall I proceed?

User:  yes
Agent: That card was rejected — the card number failed validation.
       Please double-check the number and try again.

User:  4532015112830366
Agent: Got it. To confirm, paying ₹200 using the card ending in 0366
       (CVV, expiry, and name from before). Shall I proceed?

User:  yes
Agent: Payment of ₹200 processed successfully. Transaction ID: txn_...
```

### Edge case: leap year

```
User:  ACC1004
Agent: To verify your identity...

User:  Rahul Mehta, DOB 1989-02-29
Agent: That's not a valid calendar date. Could you please confirm your
       date of birth?      [no attempt burned]

User:  Sorry, 1988-02-29
Agent: Verified. Your outstanding balance is ₹3,200.50. How much would
       you like to pay today?
```

---

## Project layout

```
payassist_agent/
├── agent.py                  # Agent class + function-calling loop (110 lines)
├── prompts.py                # System prompt (~3.8k tokens)
├── agent_tools.py            # 5 LLM-facing tools + Session dataclass
├── api_client.py             # POST /lookup-account, /process-payment
├── validators.py             # Luhn, leap year, formats
├── server.py                 # FastAPI + SSE log streaming
├── static/index.html         # Web UI (single file, vanilla)
├── cli.py                    # Terminal REPL
├── eval.py                   # 22 scenarios + ~65 assertions
├── baseline_4o_full.json     # Canonical eval baseline at submission
├── requirements.txt
└── README.md
```

---

## What I'd do with more time

- **Streaming responses** for better perceived latency on slow tool sequences
- **Prompt caching** measurement — at ~3.8k tokens of system prompt, caching is a meaningful cost win on repeat conversations
- **Multi-account session** for users with several accounts — currently switching means a fresh session
- **Voice channel adaptation** — ASR/TTS-friendly response shaping (numbers as words, shorter sentences, no punctuation cues lost in audio). The Prodigal JD specifically mentions voice agents.
- **Fine-tune a smaller model** on logged production conversations once enough volume exists, dropping latency and cost while keeping behavior
- **Observability via LangSmith or similar** — currently we log structured tool calls, which is enough for the eval, but production would want full trace storage and search

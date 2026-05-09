"""FastAPI server — wraps the Agent class in HTTP endpoints."""
from __future__ import annotations
import uuid, logging, queue, json, os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import Agent

# ── Log broadcasting ───────────────────────────────────────────────────
_log_queue: queue.Queue = queue.Queue(maxsize=500)


class UILogHandler(logging.Handler):
    """Pushes log records to the browser via SSE."""
    def emit(self, record):
        msg = self.format(record)
        kind = "info"
        if   "USER  →"   in msg: kind = "user"
        elif "AGENT →"   in msg: kind = "agent"
        elif "[TOOL]"    in msg: kind = "llm"
        elif "[API]"     in msg: kind = "api"
        elif "[VERIFY]"  in msg: kind = "state"
        elif "SESSION"   in msg: kind = "session"
        elif "ERROR" in msg or "error" in msg.lower(): kind = "error"
        try:
            _log_queue.put_nowait({
                "time": datetime.now().strftime("%H:%M:%S"),
                "kind": kind,
                "msg":  msg.strip(),
            })
        except queue.Full:
            pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("payassist")
ui_handler = UILogHandler()
ui_handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(ui_handler)


app = FastAPI(title="PayAssist API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_sessions: dict[str, Agent] = {}


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    message: str


@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    agent = Agent()
    response = agent.next("")  # opening turn → greeting
    _sessions[session_id] = agent
    log.info(f"━━ NEW SESSION [{session_id[:8]}] ━━")
    return {"session_id": session_id, "message": response["message"]}


@app.post("/chat", response_model=MessageResponse)
def chat(req: MessageRequest):
    agent = _sessions.get(req.session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found.")

    log.info(f'USER  → "{req.message}"')
    response = agent.next(req.message)
    log.info(f'AGENT → "{response["message"][:140]}"')
    log.info("─" * 40)

    return MessageResponse(session_id=req.session_id, message=response["message"])


@app.get("/logs")
async def stream_logs():
    def event_stream():
        yield 'data: {}\n\n'
        while True:
            try:
                entry = _log_queue.get(timeout=30)
                yield f"data: {json.dumps(entry)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "ok"}


@app.get("/health")
def health():
    from agent import MODEL
    return {"status": "ok", "sessions": len(_sessions), "model": MODEL}


# Mount static UI if present (no error if missing)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return FileResponse("static/index.html")
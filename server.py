"""FastAPI backend for MemoryAgent.

Wraps the existing retrieve → respond → reflect agent behind a small HTTP API
and serves the single-page web UI. One MemoryAgent is kept per session id
(all sessions share the same persistent Chroma store, exactly like the CLI).

Run:
    pip install -r requirements.txt
    uvicorn server:app --reload
    # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memoryagent.agent import MemoryAgent
from memoryagent.config import DEFAULT_LLM
from memoryagent.memory import scoring
from memoryagent.memory.schema import now

app = FastAPI(title="MemoryAgent API", version="1.0")

# ------------------------------------------------------------------ agents
# MemoryAgent keeps in-process conversation state, so we cache one per session.
# A per-session lock serialises turns (each agent mutates shared history/store).
_agents: Dict[str, MemoryAgent] = {}
_locks: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get(session: str):
    session = (session or "default").strip() or "default"
    with _registry_lock:
        if session not in _agents:
            _agents[session] = MemoryAgent(session_id=session)
            _locks[session] = threading.Lock()
        return _agents[session], _locks[session]


def _recall_view(agent: MemoryAgent) -> List[dict]:
    """Serialise the last-turn recall + its ranking breakdown for /why."""
    out = []
    for mem, bd in agent.last_recall:
        out.append({
            "type": mem.type,
            "content": mem.content,
            "score": round(float(bd.get("score", 0.0)), 3),
            "breakdown": {k: round(float(v), 3) for k, v in bd.items() if k != "score"},
        })
    return out


# ------------------------------------------------------------------ schemas
class ChatIn(BaseModel):
    session: str = "default"
    message: str


# ------------------------------------------------------------------ routes
@app.get("/api/status")
def status():
    return {
        "offline": DEFAULT_LLM.offline,
        "mode": "offline stub" if DEFAULT_LLM.offline else f"Qwen ({DEFAULT_LLM.chat_model})",
        "chat_model": None if DEFAULT_LLM.offline else DEFAULT_LLM.chat_model,
        "sessions": sorted(_agents.keys()),
    }


@app.post("/api/chat")
def chat(body: ChatIn):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "empty message")
    agent, lock = _get(body.session)
    with lock:
        reply = agent.chat(msg)
        recalled = _recall_view(agent)
        count = agent.store.count()
    return {"reply": reply, "recalled": recalled, "memory_count": count}


@app.get("/api/memories")
def memories(session: str = "default"):
    agent, lock = _get(session)
    with lock:
        rows = agent.memory_snapshot()
        count = agent.store.count()
    return {"memories": rows, "count": count}


@app.get("/api/why")
def why(session: str = "default"):
    agent, _ = _get(session)
    return {"recalled": _recall_view(agent)}


@app.post("/api/forget")
def forget(session: str = "default"):
    agent, lock = _get(session)
    with lock:
        gone = agent.forget_now()
    return {"forgotten": [{"type": m.type, "content": m.content} for m in gone]}


@app.post("/api/compress")
def compress(session: str = "default"):
    agent, lock = _get(session)
    with lock:
        result = agent.compress_memories()
    return {"result": result}


@app.post("/api/clear")
def clear(session: str = "default"):
    agent, lock = _get(session)
    with lock:
        agent.store.clear()
        agent.history = []
        agent.last_recall = []
    return {"ok": True}


# ------------------------------------------------------------------ static UI
_WEB = os.path.join(os.path.dirname(__file__), "web")


@app.get("/")
def index():
    return FileResponse(os.path.join(_WEB, "index.html"))


if os.path.isdir(_WEB):
    app.mount("/static", StaticFiles(directory=_WEB), name="static")

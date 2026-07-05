"""LLM + embedding access for Qwen Cloud, with an offline fallback.

If DASHSCOPE_API_KEY is set we talk to Qwen via the OpenAI-compatible
DashScope endpoint. Otherwise we fall back to a deterministic local embedder
and a stub chat model so the whole agent (and the demo) still runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import List

from .config import LLMConfig, DEFAULT_LLM


class LLM:
    def __init__(self, cfg: LLMConfig = DEFAULT_LLM):
        self.cfg = cfg
        self._client = None
        if not cfg.offline:
            from openai import OpenAI

            self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    # ------------------------------------------------------------------ chat
    def chat(self, messages: List[dict], temperature: float | None = None) -> str:
        if self._client is None:
            return _stub_chat(messages)
        resp = self._client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content.strip()

    def chat_json(self, messages: List[dict]) -> dict | list:
        """Chat call that must return JSON. Robust to code-fenced output."""
        if self._client is None:
            return _stub_extract(messages)
        resp = self._client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return _loads_loose(resp.choices[0].message.content)

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._client is None:
            return [_local_embed(t, self.cfg.embed_dim) for t in texts]
        resp = self._client.embeddings.create(
            model=self.cfg.embed_model,
            input=texts,
            dimensions=self.cfg.embed_dim,
        )
        return [d.embedding for d in resp.data]


# --------------------------------------------------------------------------- #
# Offline fallbacks
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _local_embed(text: str, dim: int) -> List[float]:
    """Deterministic hashing bag-of-words embedding.

    Not semantically deep, but stable and offline: lexically overlapping texts
    land near each other, which is enough to exercise retrieval end-to-end.
    """
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _stub_chat(messages: List[dict]) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    mem = next((m["content"] for m in messages if m["role"] == "system"
                and "RELEVANT MEMORIES" in m["content"]), "")
    recalled = "\n".join(l for l in mem.splitlines() if l.startswith("- ")) or "  (none)"
    return (
        "[offline stub — set DASHSCOPE_API_KEY for real Qwen responses]\n"
        f"You said: {user}\n"
        f"Grounded in memory:\n{recalled}"
    )


def _stub_extract(messages: List[dict]) -> dict:
    """Heuristic memory extraction for offline mode.

    Only inspects the USER portion of the reflection payload (never the
    assistant echo), and grades importance so ephemeral/temporal statements
    stay low and get forgotten while preferences/identity are sticky.
    """
    text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    # Payload is "USER MESSAGE:\n...\n\nASSISTANT REPLY:\n..." — keep user part.
    text = text.split("USER MESSAGE:", 1)[-1].split("ASSISTANT REPLY:", 1)[0].strip()

    EPHEMERAL = ("today", "right now", "currently", "this morning",
                 "this afternoon", "tonight", "raining", "weather")
    PREF = ("i prefer", "prefer ", "i like", "i love", "i hate",
            "always ", "never ", "please always")
    IDENTITY = ("i'm ", "i am ", "my name", "call me")

    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in EPHEMERAL):
            out.append({"content": s, "type": "event", "importance": 0.25})
        elif any(k in low for k in PREF):
            out.append({"content": s, "type": "preference", "importance": 0.85})
        elif any(k in low for k in IDENTITY):
            out.append({"content": s, "type": "profile", "importance": 0.85})
    return {"memories": out}


def _loads_loose(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[].*[\}\]]", text, re.S)
        return json.loads(m.group(0)) if m else {"memories": []}

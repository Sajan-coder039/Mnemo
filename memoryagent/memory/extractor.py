"""Turn a conversation turn into durable memories (reflection).

After each exchange the agent reflects: what here is worth remembering later?
We ask the LLM for atomic, self-contained memories with a type and an
importance in 0..1. Ephemeral chit-chat should yield nothing.
"""
from __future__ import annotations

from typing import List

from ..llm import LLM
from .schema import Memory, MEMORY_TYPES

_SYSTEM = (
    "You extract durable memories from a conversation so an assistant can serve "
    "this user better in future sessions. Return JSON: "
    '{"memories": [{"content": str, "type": str, "importance": float}]}.\n'
    f"type must be one of: {', '.join(MEMORY_TYPES)}.\n"
    "Guidelines:\n"
    "- Store stable facts, preferences, identity, and standing instructions.\n"
    "- Each memory must be atomic and self-contained (resolve pronouns; no 'it'/'that').\n"
    "- importance: preferences/identity/standing-instructions ~0.8-0.95; "
    "useful facts ~0.5-0.7; one-off/ephemeral details ~0.2-0.4.\n"
    "- Do NOT store greetings, small talk, or things true only for this moment.\n"
    "- Return an empty list if nothing is worth remembering."
)


class Extractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, user_msg: str, assistant_msg: str, session_id: str) -> List[Memory]:
        payload = (
            f"USER MESSAGE:\n{user_msg}\n\n"
            f"ASSISTANT REPLY:\n{assistant_msg}\n\n"
            "Extract memories as JSON."
        )
        data = self.llm.chat_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": payload}]
        )
        items = data.get("memories", []) if isinstance(data, dict) else []
        mems: List[Memory] = []
        for it in items:
            content = (it.get("content") or "").strip()
            if not content:
                continue
            mems.append(
                Memory.create(
                    content=content,
                    type=it.get("type", "fact"),
                    importance=float(it.get("importance", 0.5)),
                    session_id=session_id,
                    source="user",
                )
            )
        return mems

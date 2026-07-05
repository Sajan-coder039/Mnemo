"""The Memory record and its (de)serialization to Chroma metadata."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

# Memory categories. `preference` / `profile` / `instruction` are the sticky,
# high-value kinds; `fact` / `event` are more ephemeral by default.
MEMORY_TYPES = ("preference", "profile", "instruction", "fact", "event")


def now() -> float:
    return time.time()


@dataclass
class Memory:
    id: str
    content: str
    type: str
    importance: float          # 0..1, assigned at write time
    created_at: float
    last_accessed: float
    access_count: int
    session_id: str
    source: str                # user | agent | consolidation

    # ------------------------------------------------------------- factories
    @classmethod
    def create(
        cls,
        content: str,
        type: str = "fact",
        importance: float = 0.5,
        session_id: str = "default",
        source: str = "user",
    ) -> "Memory":
        t = now()
        if type not in MEMORY_TYPES:
            type = "fact"
        return cls(
            id=uuid.uuid4().hex,
            content=content.strip(),
            type=type,
            importance=max(0.0, min(1.0, float(importance))),
            created_at=t,
            last_accessed=t,
            access_count=0,
            session_id=session_id,
            source=source,
        )

    # ---------------------------------------------------- chroma (de)serialize
    def metadata(self) -> dict:
        return {
            "type": self.type,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "session_id": self.session_id,
            "source": self.source,
        }

    @classmethod
    def from_chroma(cls, id: str, document: str, meta: dict) -> "Memory":
        return cls(
            id=id,
            content=document,
            type=meta.get("type", "fact"),
            importance=float(meta.get("importance", 0.5)),
            created_at=float(meta.get("created_at", 0.0)),
            last_accessed=float(meta.get("last_accessed", 0.0)),
            access_count=int(meta.get("access_count", 0)),
            session_id=meta.get("session_id", "default"),
            source=meta.get("source", "user"),
        )

    def age_days(self, ref: Optional[float] = None) -> float:
        return max(0.0, ((ref or now()) - self.created_at) / 86400.0)

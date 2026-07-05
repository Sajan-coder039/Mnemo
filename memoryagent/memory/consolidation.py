"""LLM-judged consolidation: dedup-merge and summarization.

Pure cosine dedup misses paraphrases ("User prefers Python" vs "strongly
prefers Python over JavaScript") that sit in a gray similarity band. Here an
LLM decides whether two memories are the same fact and, if so, produces one
canonical phrasing. The same component compresses many aged, low-value memories
into a few durable notes.

Both methods degrade gracefully offline (token-overlap heuristics).
"""
from __future__ import annotations

import re
from typing import List

from ..llm import LLM

_TOK = re.compile(r"[a-z0-9']+")


def _tokens(s: str) -> set:
    return set(_TOK.findall(s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Consolidator:
    def __init__(self, llm: LLM):
        self.llm = llm

    # --------------------------------------------------------- dedup / merge
    def judge_merge(self, existing: str, incoming: str) -> dict:
        """Decide if two memories state the same fact. Returns
        {"same": bool, "merged": str} where merged is the canonical phrasing."""
        if self.llm.cfg.offline:
            same = _jaccard(existing, incoming) >= 0.5
            merged = existing if len(existing) >= len(incoming) else incoming
            return {"same": same, "merged": merged}

        sys = (
            "You decide whether two memory statements about a user express the "
            "SAME underlying fact/preference (even if phrased differently or one "
            "is more specific). Return JSON "
            '{"same": bool, "merged": str}. If same, "merged" is a single '
            "canonical statement capturing the most complete, specific version. "
            "If not same, set same=false and merged to an empty string."
        )
        user = f"A: {existing}\nB: {incoming}"
        data = self.llm.chat_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        )
        if not isinstance(data, dict):
            return {"same": False, "merged": ""}
        return {"same": bool(data.get("same")), "merged": (data.get("merged") or "").strip()}

    # ---------------------------------------------------------- summarization
    def summarize(self, contents: List[str]) -> List[str]:
        """Compress many aged, low-value memories into a few durable notes."""
        if not contents:
            return []
        if self.llm.cfg.offline:
            return ["Earlier context: " + "; ".join(c.rstrip(".") for c in contents) + "."]

        sys = (
            "You compress a list of older, low-importance user memories into at "
            "most 2 concise, durable notes, keeping only information likely to "
            "matter later and dropping the purely transient. Return JSON "
            '{"notes": [str, ...]}. Return an empty list if nothing is worth keeping.'
        )
        user = "MEMORIES:\n" + "\n".join(f"- {c}" for c in contents)
        data = self.llm.chat_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        )
        notes = data.get("notes", []) if isinstance(data, dict) else []
        return [n.strip() for n in notes if isinstance(n, str) and n.strip()]

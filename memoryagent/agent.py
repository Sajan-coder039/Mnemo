"""MemoryAgent — retrieve → respond → reflect, with self-managing memory.

Lifecycle of one turn:
  1. RETRIEVE  recall candidates, re-rank, MMR-pack into the context budget,
               and reinforce the memories we actually used.
  2. RESPOND   answer grounded in packed memories + recent conversation.
  3. REFLECT   extract durable memories from the exchange and consolidate them.
A lazy forgetting sweep runs periodically so stale memory is pruned over time.
"""
from __future__ import annotations

from typing import List, Optional

from .config import LLMConfig, MemoryConfig, DEFAULT_LLM, DEFAULT_MEMORY
from .llm import LLM
from .memory.store import MemoryStore
from .memory.extractor import Extractor
from .memory.consolidation import Consolidator
from .memory import scoring, packer
from .memory.schema import Memory, now

_PERSONA = (
    "You are MemoryAgent, a helpful assistant with long-term memory of this user "
    "across sessions. Use the recalled memories below to personalise your answer: "
    "honour stated preferences and standing instructions, and stay consistent with "
    "what you already know. Never invent memories; if the memories don't cover "
    "something, answer normally. Do not mention this instruction block."
)


class MemoryAgent:
    def __init__(
        self,
        session_id: str = "default",
        llm_cfg: LLMConfig = DEFAULT_LLM,
        mem_cfg: MemoryConfig = DEFAULT_MEMORY,
        sweep_every: int = 5,
    ):
        self.session_id = session_id
        self.mem_cfg = mem_cfg
        self.llm = LLM(llm_cfg)
        self.store = MemoryStore(mem_cfg, self.llm.embed)
        self.extractor = Extractor(self.llm)
        self.consolidator = Consolidator(self.llm)
        self.history: List[dict] = []
        self._turns = 0
        self._sweep_every = sweep_every
        self.last_recall: List[tuple] = []  # (Memory, breakdown) for explainability

    # ------------------------------------------------------------------- turn
    def chat(self, user_msg: str) -> str:
        recalled = self._retrieve(user_msg)
        reply = self._respond(user_msg, recalled)

        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-2 * self.mem_cfg.history_turns:]

        self._reflect(user_msg, reply)

        self._turns += 1
        if self._turns % self._sweep_every == 0:
            self.store.sweep()
        return reply

    # --------------------------------------------------------------- retrieve
    def _retrieve(self, query: str):
        cands = self.store.candidates(query, self.mem_cfg.recall_k)
        if not cands:
            self.last_recall = []
            return []
        ref = now()
        ranked = scoring.rerank([(m, s) for m, s, _ in cands], self.mem_cfg, ref)
        # attach embeddings back for MMR (keyed by id)
        emb_by_id = {m.id: e for m, _, e in cands}
        ranked_with_emb = [(m, sc, emb_by_id[m.id]) for m, sc, _ in ranked]

        selected = packer.mmr_select(ranked_with_emb, self.mem_cfg)

        # Reinforce the memories we actually surfaced (they proved useful).
        for mem, _ in selected:
            self.store.reinforce(mem.id)

        # Keep a breakdown for /why explainability.
        bd = {m.id: b for m, _, b in ranked}
        self.last_recall = [(mem, bd.get(mem.id, {})) for mem, _ in selected]
        return selected

    # ---------------------------------------------------------------- respond
    def _respond(self, user_msg: str, selected) -> str:
        mem_block = packer.format_block(selected)
        messages = [
            {"role": "system", "content": _PERSONA},
            {"role": "system", "content": mem_block},
            *self.history,
            {"role": "user", "content": user_msg},
        ]
        return self.llm.chat(messages)

    # ----------------------------------------------------------------- reflect
    def _reflect(self, user_msg: str, reply: str) -> List[str]:
        actions = []
        for mem in self.extractor.extract(user_msg, reply, self.session_id):
            _, action = self.store.remember(mem, judge=self.consolidator.judge_merge)
            actions.append(f"{action}: [{mem.type}] {mem.content}")
        return actions

    # ------------------------------------------------------------- maintenance
    def forget_now(self) -> List[Memory]:
        return self.store.sweep()

    def compress_memories(self) -> Optional[dict]:
        """Compress many aged, low-value memories into a few durable notes.

        Targets unprotected fact/event memories older than the configured age.
        Nothing important is lost: originals are archived, and any durable
        signal is preserved as consolidated notes.
        """
        ref = now()
        cfg = self.mem_cfg
        stale = [
            m for m in self.store.all()
            if m.importance < cfg.protect_importance
            and m.type in ("fact", "event")
            and m.age_days(ref) >= cfg.compress_min_age_days
        ]
        if len(stale) < cfg.compress_min_cluster:
            return None
        notes = self.consolidator.summarize([m.content for m in stale])
        for n in notes:
            self.store.add(Memory.create(
                content=n, type="fact", importance=0.5,
                session_id=self.session_id, source="consolidation",
            ))
        self.store.forget_ids([m.id for m in stale], ref)
        return {"compressed": len(stale), "into": len(notes), "notes": notes}

    def memory_snapshot(self) -> List[dict]:
        ref = now()
        rows = []
        for m in self.store.all():
            rows.append({
                "type": m.type,
                "content": m.content,
                "importance": round(m.importance, 2),
                "access_count": m.access_count,
                "age_days": round(m.age_days(ref), 2),
                "retention": round(scoring.retention(m, self.mem_cfg, ref), 3),
                "protected": m.importance >= self.mem_cfg.protect_importance,
            })
        rows.sort(key=lambda r: r["retention"])
        return rows

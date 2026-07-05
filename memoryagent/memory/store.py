"""Chroma-backed memory store with reinforcement, consolidation and archival.

Vectors + metadata live in a persistent Chroma collection (cosine space).
Forgotten memories are not destroyed: they are appended to an archive JSONL so
the forgetting behaviour is auditable and (in principle) reversible.
"""
from __future__ import annotations

import json
import os
from typing import Callable, List, Optional, Tuple

# A judge takes (existing_content, incoming_content) and returns
# {"same": bool, "merged": str}. Injected so the store stays LLM-agnostic.
Judge = Callable[[str, str], dict]

import numpy as np

from ..config import MemoryConfig
from .schema import Memory, now
from . import scoring


class MemoryStore:
    def __init__(self, cfg: MemoryConfig, embed_fn: Callable[[List[str]], List[List[float]]]):
        import chromadb

        self.cfg = cfg
        self.embed = embed_fn
        os.makedirs(cfg.store_path, exist_ok=True)
        self._archive_path = os.path.join(cfg.store_path, "memory_archive.jsonl")
        self._client = chromadb.PersistentClient(path=os.path.join(cfg.store_path, "chroma"))
        self._col = self._client.get_or_create_collection(
            name="memories", metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------- write
    def add(self, mem: Memory, embedding: Optional[List[float]] = None) -> Memory:
        emb = embedding if embedding is not None else self.embed([mem.content])[0]
        self._col.add(
            ids=[mem.id],
            embeddings=[emb],
            documents=[mem.content],
            metadatas=[mem.metadata()],
        )
        return mem

    def remember(self, mem: Memory, judge: Optional[Judge] = None) -> Tuple[Memory, str]:
        """Add with consolidation. Returns (resulting_memory, action) where
        action ∈ {"added", "reinforced", "merged"}.

        - cosine ≥ dedup_threshold        → auto-reinforce the existing memory.
        - cosine in [gray_low, threshold) → ask `judge`; if same fact, merge into
          one canonical phrasing and reinforce.
        - otherwise                        → store as new.
        """
        emb = self.embed([mem.content])[0]
        dup = self._nearest(emb)
        if dup is not None:
            existing, sim = dup
            if sim >= self.cfg.dedup_threshold:
                self.reinforce(existing.id, bump_importance=max(existing.importance, mem.importance))
                return existing, "reinforced"
            if judge is not None and sim >= self.cfg.dedup_gray_low:
                verdict = judge(existing.content, mem.content)
                if verdict.get("same"):
                    merged = verdict.get("merged") or existing.content
                    if merged.strip() and merged.strip() != existing.content.strip():
                        self.update_content(existing.id, merged.strip())
                    self.reinforce(existing.id, bump_importance=max(existing.importance, mem.importance))
                    return existing, "merged"
        self.add(mem, embedding=emb)
        return mem, "added"

    def update_content(self, mem_id: str, new_content: str) -> None:
        """Replace a memory's text and re-embed it, preserving its metadata."""
        rec = self._col.get(ids=[mem_id], include=["metadatas"])
        if not rec["ids"]:
            return
        emb = self.embed([new_content])[0]
        self._col.update(
            ids=[mem_id], embeddings=[emb], documents=[new_content], metadatas=rec["metadatas"]
        )

    # --------------------------------------------------------------- retrieval
    def candidates(self, query: str, k: int) -> List[Tuple[Memory, float, List[float]]]:
        """Stage-1 recall: ANN nearest neighbours with cosine similarity."""
        if self.count() == 0:
            return []
        q_emb = self.embed([query])[0]
        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        out = []
        ids = res["ids"][0]
        for i, mid in enumerate(ids):
            mem = Memory.from_chroma(mid, res["documents"][0][i], res["metadatas"][0][i])
            # Chroma cosine distance = 1 - cosine_similarity
            sim = 1.0 - float(res["distances"][0][i])
            emb = res["embeddings"][0][i]
            out.append((mem, max(0.0, sim), list(emb)))
        return out

    def _nearest(self, embedding: List[float]) -> Optional[Tuple[Memory, float]]:
        if self.count() == 0:
            return None
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if not res["ids"][0]:
            return None
        mem = Memory.from_chroma(res["ids"][0][0], res["documents"][0][0], res["metadatas"][0][0])
        sim = 1.0 - float(res["distances"][0][0])
        return mem, sim

    # ----------------------------------------------------------- reinforcement
    def reinforce(self, mem_id: str, bump_importance: Optional[float] = None) -> None:
        rec = self._col.get(ids=[mem_id], include=["metadatas"])
        if not rec["ids"]:
            return
        meta = dict(rec["metadatas"][0])
        meta["last_accessed"] = now()
        meta["access_count"] = int(meta.get("access_count", 0)) + 1
        if bump_importance is not None:
            meta["importance"] = max(float(meta.get("importance", 0.0)), float(bump_importance))
        self._col.update(ids=[mem_id], metadatas=[meta])

    # -------------------------------------------------------------- forgetting
    def sweep(self, ref: Optional[float] = None) -> List[Memory]:
        """Forget memories whose retention has decayed below threshold.

        Forgotten memories are archived to JSONL, then removed from Chroma.
        Returns the list of forgotten memories.
        """
        ref = ref or now()
        forgotten = []
        for mem in self.all():
            if scoring.should_forget(mem, self.cfg, ref):
                forgotten.append(mem)
        if forgotten:
            self._archive(forgotten, ref)
            self._col.delete(ids=[m.id for m in forgotten])
        return forgotten

    def forget_ids(self, ids: List[str], ref: Optional[float] = None) -> List[Memory]:
        """Archive + delete specific memories (used by summarization/compaction)."""
        ref = ref or now()
        idset = set(ids)
        mems = [m for m in self.all() if m.id in idset]
        if mems:
            self._archive(mems, ref)
            self._col.delete(ids=[m.id for m in mems])
        return mems

    def _archive(self, mems: List[Memory], ref: float) -> None:
        with open(self._archive_path, "a") as f:
            for m in mems:
                row = m.metadata()
                row.update({
                    "id": m.id,
                    "content": m.content,
                    "forgotten_at": ref,
                    "retention_at_forget": round(scoring.retention(m, self.cfg, ref), 4),
                })
                f.write(json.dumps(row) + "\n")

    # -------------------------------------------------------------- inspection
    def all(self) -> List[Memory]:
        rec = self._col.get(include=["documents", "metadatas"])
        return [
            Memory.from_chroma(mid, rec["documents"][i], rec["metadatas"][i])
            for i, mid in enumerate(rec["ids"])
        ]

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)

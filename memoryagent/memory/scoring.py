"""Retrieval ranking and forgetting — the core algorithms.

Retrieval uses a two-stage design:
  1. Chroma returns the top-K nearest candidates by cosine similarity (recall).
  2. Those candidates are re-ranked by a composite score that blends semantic
     similarity with recency, importance, and access frequency (precision).

Forgetting uses an Ebbinghaus-style retention curve. Each memory has a
`stability` that grows with its importance and with every time it is recalled
(reinforcement). Retention decays exponentially with time since last access;
when it drops below a threshold the memory is forgotten — unless it is flagged
important enough to be protected.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from ..config import MemoryConfig
from .schema import Memory, now


# --------------------------------------------------------------------------- #
# Component scores (all normalised to 0..1)
# --------------------------------------------------------------------------- #
def recency_score(mem: Memory, cfg: MemoryConfig, ref: float) -> float:
    dt_days = max(0.0, (ref - mem.last_accessed) / 86400.0)
    return 0.5 ** (dt_days / cfg.recency_half_life_days)


def frequency_score(mem: Memory, cfg: MemoryConfig) -> float:
    # Saturating: diminishing returns for very frequently used memories.
    return 1.0 - math.exp(-mem.access_count / cfg.frequency_k)


def composite_score(
    mem: Memory, semantic: float, cfg: MemoryConfig, ref: float | None = None
) -> Tuple[float, dict]:
    """Return the final ranking score and a breakdown (for explainability)."""
    ref = ref or now()
    rec = recency_score(mem, cfg, ref)
    freq = frequency_score(mem, cfg)
    imp = mem.importance
    w = cfg.weights
    score = (
        w.semantic * semantic
        + w.recency * rec
        + w.importance * imp
        + w.frequency * freq
    )
    return score, {
        "semantic": round(semantic, 3),
        "recency": round(rec, 3),
        "importance": round(imp, 3),
        "frequency": round(freq, 3),
        "score": round(score, 3),
    }


# --------------------------------------------------------------------------- #
# Forgetting
# --------------------------------------------------------------------------- #
def retention(mem: Memory, cfg: MemoryConfig, ref: float | None = None) -> float:
    """Ebbinghaus retention in 0..1.

    stability = base * (0.5 + importance) * (1 + ln(1 + access_count))
    retention = 0.5 ** (days_since_access / stability)
    """
    ref = ref or now()
    dt_days = max(0.0, (ref - mem.last_accessed) / 86400.0)
    stability = (
        cfg.retention_base_days
        * (0.5 + mem.importance)
        * (1.0 + math.log1p(mem.access_count))
    )
    return 0.5 ** (dt_days / stability)


def should_forget(mem: Memory, cfg: MemoryConfig, ref: float | None = None) -> bool:
    if mem.importance >= cfg.protect_importance:
        return False  # protected: preferences, profile, key instructions
    return retention(mem, cfg, ref) < cfg.forget_threshold


def rerank(
    candidates: List[Tuple[Memory, float]], cfg: MemoryConfig, ref: float | None = None
) -> List[Tuple[Memory, float, dict]]:
    """candidates: (memory, semantic_similarity) → sorted (memory, score, breakdown)."""
    ref = ref or now()
    scored = []
    for mem, sem in candidates:
        score, breakdown = composite_score(mem, sem, cfg, ref)
        scored.append((mem, score, breakdown))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

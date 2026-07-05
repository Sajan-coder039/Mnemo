"""Context-window packing.

Given re-ranked memories and a token budget, select a subset that is both
relevant and non-redundant (Maximal Marginal Relevance) and format it compactly
for injection into the prompt. This is where "recall the critical memories
within a limited context window" happens.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import MemoryConfig
from .schema import Memory


def _tokens(text: str) -> int:
    # Cheap, provider-agnostic estimate (~4 chars/token).
    return max(1, len(text) // 4)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    ranked: List[Tuple[Memory, float, List[float]]],
    cfg: MemoryConfig,
) -> List[Tuple[Memory, float]]:
    """MMR over the re-ranked candidates, bounded by the token budget.

    ranked: (memory, composite_score, embedding), pre-sorted by score desc.
    Returns selected (memory, score) in selection order.
    """
    if not ranked:
        return []
    lam = cfg.mmr_lambda
    embs = [np.asarray(e, dtype=float) for _, _, e in ranked]
    scores = [s for _, s, _ in ranked]

    selected: List[int] = []
    selected_out: List[Tuple[Memory, float]] = []
    used_tokens = 0
    candidate_idx = list(range(len(ranked)))

    while candidate_idx and len(selected_out) < cfg.return_k:
        best_i, best_val = None, -1e9
        for i in candidate_idx:
            if selected:
                max_sim = max(_cos(embs[i], embs[j]) for j in selected)
            else:
                max_sim = 0.0
            mmr = lam * scores[i] - (1.0 - lam) * max_sim
            if mmr > best_val:
                best_val, best_i = mmr, i

        mem = ranked[best_i][0]
        cost = _tokens(mem.content)
        candidate_idx.remove(best_i)
        if used_tokens + cost > cfg.context_token_budget and selected_out:
            continue  # skip; try to fit smaller memories in remaining budget
        used_tokens += cost
        selected.append(best_i)
        selected_out.append((mem, scores[best_i]))

    return selected_out


def format_block(selected: List[Tuple[Memory, float]]) -> str:
    """Render selected memories grouped by type for the system prompt."""
    if not selected:
        return "RELEVANT MEMORIES: (none)"
    by_type: dict[str, list] = {}
    for mem, _ in selected:
        by_type.setdefault(mem.type, []).append(mem)
    lines = ["RELEVANT MEMORIES (recalled for this turn):"]
    for typ in ("profile", "preference", "instruction", "fact", "event"):
        if typ not in by_type:
            continue
        for mem in by_type[typ]:
            lines.append(f"- [{typ}] {mem.content}")
    return "\n".join(lines)

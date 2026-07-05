"""Retrieval eval harness for MemoryAgent.

Seeds a labelled memory set, runs a set of queries, and reports precision@k,
recall@k and MRR — comparing the composite re-ranker against a pure-semantic
baseline. This makes the ranking design measurable and tunable rather than
assumed.

    python eval_retrieval.py            # offline (local embedder)
    DASHSCOPE_API_KEY=... python eval_retrieval.py   # real Qwen embeddings
"""
from __future__ import annotations

import os
import shutil
import tempfile

from memoryagent.config import LLMConfig, MemoryConfig
from memoryagent.llm import LLM
from memoryagent.memory.store import MemoryStore
from memoryagent.memory.schema import Memory
from memoryagent.memory import scoring

K = 3  # cutoff for @k metrics

# key → (content, type, importance). key is only used for relevance labelling.
MEMORIES = {
    "veg":      ("The user is vegetarian and avoids meat.", "preference", 0.9),
    "python":   ("The user strongly prefers Python over JavaScript.", "preference", 0.9),
    "concise":  ("The user wants concise, bullet-point answers.", "preference", 0.85),
    "name":     ("The user's name is Nirajan.", "profile", 0.95),
    "coffee":   ("The user drinks black coffee, no sugar.", "preference", 0.6),
    "pytest":   ("The user uses pytest for testing Python code.", "fact", 0.6),
    "tabs":     ("The user indents code with 4 spaces, never tabs.", "preference", 0.7),
    "macos":    ("The user works on macOS.", "fact", 0.5),
    "postgres": ("The user's team uses PostgreSQL as their database.", "fact", 0.6),
    "gym":      ("The user goes to the gym on weekday mornings.", "fact", 0.4),
    "dog":      ("The user has a dog named Momo.", "fact", 0.4),
    "spicy":    ("The user dislikes very spicy food.", "preference", 0.6),
}

# query → set of relevant keys
QUERIES = [
    ("What should I make for dinner tonight?", {"veg", "spicy"}),
    ("Which language should I write this quick script in?", {"python"}),
    ("How do you want me to format my answers?", {"concise"}),
    ("Set up a testing framework for my code.", {"pytest", "python"}),
    ("What database should the backend connect to?", {"postgres"}),
    ("Remind me what my name is.", {"name"}),
    ("Recommend a coffee order.", {"coffee"}),
]


def seed(store: MemoryStore):
    key_by_content = {}
    for _, (content, typ, imp) in MEMORIES.items():
        m = Memory.create(content=content, type=typ, importance=imp, session_id="eval")
        store.add(m)
        key_by_content[content] = _
    # map content back to key for scoring
    return {v[0]: k for k, v in MEMORIES.items()}


def rank(store, query, mode):
    cands = store.candidates(query, k=len(MEMORIES))
    if mode == "semantic":
        ordered = sorted(cands, key=lambda c: c[1], reverse=True)
        return [m for m, _, _ in ordered]
    # composite
    ranked = scoring.rerank([(m, s) for m, s, _ in cands], store.cfg)
    return [m for m, _, _ in ranked]


def metrics(retrieved_keys, relevant):
    topk = retrieved_keys[:K]
    hits = [k for k in topk if k in relevant]
    precision = len(hits) / K
    recall = len(hits) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for i, k in enumerate(retrieved_keys):
        if k in relevant:
            mrr = 1.0 / (i + 1)
            break
    return precision, recall, mrr


def main():
    path = tempfile.mkdtemp(prefix="memoryagent-eval-")
    llm = LLM(LLMConfig(api_key=os.environ.get("DASHSCOPE_API_KEY", "")))
    cfg = MemoryConfig(store_path=path)
    mode_label = "Qwen embeddings" if not llm.cfg.offline else "offline local embedder"
    print(f"Retrieval eval — {mode_label}, k={K}, {len(MEMORIES)} memories, "
          f"{len(QUERIES)} queries\n")

    try:
        store = MemoryStore(cfg, llm.embed)
        content_to_key = seed(store)

        agg = {"semantic": [0, 0, 0], "composite": [0, 0, 0]}
        print(f"{'query':<48} {'semantic P/R/MRR':<22} composite P/R/MRR")
        print("-" * 92)
        for q, relevant in QUERIES:
            row = f"{q[:46]:<48} "
            for mode in ("semantic", "composite"):
                mems = rank(store, q, mode)
                keys = [content_to_key[m.content] for m in mems]
                p, r, mrr = metrics(keys, relevant)
                agg[mode][0] += p
                agg[mode][1] += r
                agg[mode][2] += mrr
                row += f"{p:.2f}/{r:.2f}/{mrr:.2f}".ljust(22 if mode == "semantic" else 0)
            print(row)

        n = len(QUERIES)
        print("-" * 92)
        for mode in ("semantic", "composite"):
            p, r, mrr = (x / n for x in agg[mode])
            print(f"  {mode:<10} mean:  P@{K}={p:.3f}  R@{K}={r:.3f}  MRR={mrr:.3f}")
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()

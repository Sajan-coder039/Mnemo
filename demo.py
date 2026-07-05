"""Scripted demo of MemoryAgent: cross-session recall, forgetting, reinforcement.

Runs fully offline (local embedder + stub LLM) so no API key is needed:
    python demo.py

To see it drive real Qwen instead, export DASHSCOPE_API_KEY first.

Time is fast-forwarded by rewriting memory timestamps so decay/forgetting is
observable in seconds instead of days.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from memoryagent.config import LLMConfig, MemoryConfig
from memoryagent.agent import MemoryAgent
from memoryagent.memory.schema import now


def age_all_memories(store, days: float):
    """Fast-forward: pretend every memory was last accessed `days` ago."""
    rec = store._col.get(include=["metadatas"])
    t = now() - days * 86400.0
    metas = []
    for m in rec["metadatas"]:
        m = dict(m)
        m["last_accessed"] = t
        metas.append(m)
    if rec["ids"]:
        store._col.update(ids=rec["ids"], metadatas=metas)


def show(agent, title):
    print(f"\n─── {title} " + "─" * (60 - len(title)))
    for r in agent.memory_snapshot():
        prot = " 🔒protected" if r["protected"] else ""
        print(f"  [{r['type']:<11}] imp={r['importance']:.2f} acc={r['access_count']} "
              f"ret={r['retention']:.2f}{prot}  {r['content']}")


def main():
    store_path = tempfile.mkdtemp(prefix="memoryagent-demo-")
    llm = LLMConfig(api_key=os.environ.get("DASHSCOPE_API_KEY", ""))
    mem = MemoryConfig(store_path=store_path)
    print(f"store: {store_path}  |  mode: {'Qwen' if not llm.offline else 'offline stub'}")

    try:
        # ---------------- Session 1: user shares preferences + an aside -------
        print("\n######## SESSION 1 ########")
        a1 = MemoryAgent(session_id="s1", llm_cfg=llm, mem_cfg=mem)
        for msg in [
            "Hi! I'm Nirajan. I'm a vegetarian and I strongly prefer Python over JavaScript.",
            "Please always keep your answers concise — bullet points, no fluff.",
            # A time-bound, mid-value fact: useful now, stale later → should be
            # stored but NOT protected, so the forgetting curve can act on it.
            "For the next two weeks I'm travelling in Lisbon, so I'm on GMT+1.",
            "By the way, it's raining in my city today.",  # ephemeral → low/none
        ]:
            print(f"\nyou> {msg}")
            print(f"agent> {a1.chat(msg)}")
        show(a1, "Memories after session 1")

        # ---------------- Forgetting: fast-forward 12 days --------------------
        print("\n######## 20 DAYS LATER (forgetting sweep) ########")
        age_all_memories(a1.store, days=20)
        forgotten = a1.forget_now()
        print(f"forgotten: {len(forgotten)}")
        for m in forgotten:
            print(f"  - dropped [{m.type}] imp={m.importance:.2f}  {m.content}")
        show(a1, "Memories after forgetting sweep")
        if forgotten:
            print("  → low-importance, unprotected memories decayed past the "
                  "threshold and were archived; high-value preferences protected.")
        else:
            print("  → nothing crossed the forget threshold: everything stored "
                  "was high-importance and protected (see decayed 'ret' values).")

        # ---------------- Session 2: brand-new agent, same store --------------
        print("\n######## SESSION 2 (new process, same memory store) ########")
        a2 = MemoryAgent(session_id="s2", llm_cfg=llm, mem_cfg=mem)
        q = "Can you suggest a dinner recipe and tell me what language to write a quick script in?"
        print(f"\nyou> {q}")
        print(f"agent> {a2.chat(q)}")
        print("\nrecalled for that turn (with score breakdown):")
        for m, bd in a2.last_recall:
            print(f"  [{m.type}] {m.content}\n      {bd}")

        # ---------------- Reinforcement -------------------------------------
        show(a2, "Memories after session 2 (note bumped access counts)")
        print("  → recalling the preferences reinforced them (acc↑, recency reset).")

    finally:
        shutil.rmtree(store_path, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Demo: LLM-judged dedup-merge and memory summarization.

    python demo_consolidation.py                     # offline heuristics
    DASHSCOPE_API_KEY=... python demo_consolidation.py   # real Qwen judging

Part 1 feeds paraphrases of the same preference and shows they collapse into a
single canonical memory instead of piling up as near-duplicates.
Part 2 seeds several aged, low-value facts and compresses them into a few
durable notes.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from memoryagent.config import LLMConfig, MemoryConfig
from memoryagent.agent import MemoryAgent
from memoryagent.memory.schema import Memory, now


def show(agent, title):
    print(f"\n─── {title} " + "─" * max(0, 58 - len(title)))
    for m in agent.store.all():
        print(f"  [{m.type:<11}] imp={m.importance:.2f} acc={m.access_count} "
              f"src={m.source:<12} {m.content}")


def main():
    path = tempfile.mkdtemp(prefix="memoryagent-consol-")
    llm = LLMConfig(api_key=os.environ.get("DASHSCOPE_API_KEY", ""))
    cfg = MemoryConfig(store_path=path)
    print(f"mode: {'Qwen' if not llm.offline else 'offline heuristic'}")

    try:
        agent = MemoryAgent(session_id="c", llm_cfg=llm, mem_cfg=cfg)
        judge = agent.consolidator.judge_merge

        # ---- Part 1a: the LLM judge in isolation (deterministic) -----------
        print("\n######## PART 1a — the LLM merge-judge, directly ########")
        pairs = [
            ("I like using Python.",
             "The user's favourite programming language is Python."),
            ("The user prefers Python over other languages.",
             "User strongly prefers Python, especially over JavaScript."),
            ("The user is vegetarian.",
             "The user strongly prefers Python."),  # clearly different
        ]
        for a, b in pairs:
            v = judge(a, b)
            print(f"  same={str(v['same']):<5}  A={a!r}\n"
                  f"               B={b!r}\n"
                  f"        merged={v['merged']!r}\n")

        # ---- Part 1b: dedup-merge in the write path ------------------------
        print("######## PART 1b — dedup-merge on write ########")
        paraphrases = [
            ("I like using Python.", "preference", 0.7),
            ("The user prefers Python over other languages.", "preference", 0.8),
            ("User strongly prefers Python, especially over JavaScript.", "preference", 0.9),
            ("The user's favourite programming language is Python.", "preference", 0.75),
            ("The user is vegetarian.", "preference", 0.9),  # distinct → should stay
        ]
        for content, typ, imp in paraphrases:
            m = Memory.create(content=content, type=typ, importance=imp, session_id="c")
            _, action = agent.store.remember(m, judge=judge)
            print(f"  {action:<10} ← {content}")
        show(agent, "Store after dedup-merge")
        py = [m for m in agent.store.all() if "python" in m.content.lower()]
        print(f"  → 4 Python paraphrases written, consolidated down to "
              f"{len(py)} memory(ies); vegetarian kept separate. "
              f"(Online dedup is pairwise/order-dependent; a periodic full pass "
              f"would merge any residual.)")

        # ---- Part 2: summarization of aged low-value memories ---------------
        print("\n\n######## PART 2 — summarization of aged low-value memories ########")
        old = now() - 15 * 86400.0  # 15 days ago
        aged = [
            "The user mentioned they were debugging a flaky CI test last sprint.",
            "The user was on a call about the Q2 roadmap.",
            "The user tried a new ramen place downtown.",
            "The user was reading a paper on retrieval-augmented generation.",
        ]
        for content in aged:
            m = Memory.create(content=content, type="fact", importance=0.4, session_id="c")
            m.created_at = old
            m.last_accessed = old
            agent.store.add(m)
        show(agent, "Store before compression")

        result = agent.compress_memories()
        print(f"\n  compress result: {result}")
        show(agent, "Store after compression")
        print("  → aged low-value facts archived; durable signal kept as "
              "consolidated note(s).")
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()

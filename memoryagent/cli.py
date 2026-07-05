"""Interactive REPL for MemoryAgent.

Usage:
    python -m memoryagent.cli [--session NAME]

Commands inside the REPL:
    /mem        show all stored memories (sorted by retention, weakest first)
    /why        explain the memories recalled for the last turn
    /forget     run a forgetting sweep now
    /stats      store statistics
    /clear      wipe all memories (this session's store)
    /quit       exit
"""
from __future__ import annotations

import argparse

from .agent import MemoryAgent
from .config import DEFAULT_LLM


def _print_mem(rows):
    if not rows:
        print("  (no memories yet)")
        return
    for r in rows:
        prot = " 🔒" if r["protected"] else ""
        print(
            f"  [{r['type']:<11}] imp={r['importance']:.2f} "
            f"acc={r['access_count']:<2} age={r['age_days']:.1f}d "
            f"ret={r['retention']:.2f}{prot}  {r['content']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="default")
    args = ap.parse_args()

    agent = MemoryAgent(session_id=args.session)
    mode = "offline stub" if DEFAULT_LLM.offline else f"Qwen ({DEFAULT_LLM.chat_model})"
    print(f"MemoryAgent [{mode}] — session '{args.session}'. Type /quit to exit.")
    if DEFAULT_LLM.offline:
        print("  (no DASHSCOPE_API_KEY set — running with local embedder + stub LLM)")

    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/mem":
            _print_mem(agent.memory_snapshot())
            continue
        if msg == "/why":
            if not agent.last_recall:
                print("  (nothing recalled last turn)")
            for mem, bd in agent.last_recall:
                print(f"  [{mem.type}] {mem.content}\n      {bd}")
            continue
        if msg == "/forget":
            gone = agent.forget_now()
            print(f"  forgot {len(gone)} memories:" if gone else "  nothing to forget")
            for m in gone:
                print(f"    - [{m.type}] {m.content}")
            continue
        if msg == "/stats":
            print(f"  stored memories: {agent.store.count()}")
            continue
        if msg == "/clear":
            agent.store.clear()
            print("  memory cleared")
            continue

        print(f"\nagent> {agent.chat(msg)}")


if __name__ == "__main__":
    main()

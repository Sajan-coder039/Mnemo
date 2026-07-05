# MemoryAgent

An agent with **persistent, self-managing memory**. It accumulates experience
across turns and sessions, remembers user preferences, forgets what stops
mattering, and recalls the *critical* memories within a limited context window.

- **LLM:** Qwen Cloud (DashScope, OpenAI-compatible endpoint)
- **Embeddings:** Qwen `text-embedding-v3`
- **Vector store:** Chroma (persistent, cosine space)
- **Runs offline:** with no API key it falls back to a local embedder + stub LLM
  so the whole system (and the demo) still runs.

```
pip install -r requirements.txt

# offline (no key needed) — scripted end-to-end demo
python demo.py

# real Qwen
export DASHSCOPE_API_KEY=sk-...
# international endpoint is the default; for mainland China:
# export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python -m memoryagent.cli --session me
```

## The turn lifecycle

Each `agent.chat(user_msg)` runs three stages:

1. **RETRIEVE** — recall candidates from Chroma, re-rank them, pack them into a
   token budget, and reinforce whatever was surfaced.
2. **RESPOND** — answer, grounded in the packed memories + recent conversation.
3. **REFLECT** — extract durable memories from the exchange and consolidate them.

A **forgetting sweep** runs lazily (every N turns, and on demand) to prune
memory whose retention has decayed.

```
user ──▶ RETRIEVE ─(recall K)─▶ re-rank ─▶ MMR pack ─▶ context block
                                                          │
                                              RESPOND ◀────┘──▶ reply
                                                 │
                                              REFLECT ─▶ extract ─▶ consolidate ─▶ store
                                                                                     │
                                              periodic  ◀── forgetting sweep ────────┘
```

## Retrieval ranking (the hard part)

Two stages — cheap recall, then precise re-rank:

**Stage 1 — recall.** Chroma returns the top `recall_k` nearest memories by
cosine similarity. Fast, approximate, high-recall.

**Stage 2 — composite re-rank.** Each candidate is scored by blending four
signals, all normalized to `[0,1]` (inspired by Stanford's *Generative Agents*):

```
score = w_sem·semantic + w_rec·recency + w_imp·importance + w_freq·frequency

semantic   = cosine(query, memory)                     # relevance
recency    = 0.5 ^ (days_since_access / half_life)     # exponential decay
importance = stored 0..1 weight (assigned at write)    # salience
frequency  = 1 − e^(−access_count / k)                 # saturating usage
```

Defaults: `w_sem=0.55, w_rec=0.20, w_imp=0.15, w_freq=0.10` (all tunable in
`config.py`). The `/why` REPL command and `last_recall` expose the full
per-memory breakdown, so ranking is explainable rather than a black box.

## Forgetting (timely, not destructive)

Each memory has an **Ebbinghaus retention curve**. Its *stability* grows with
importance and with every reinforcement, so well-used and important memories
decay slowly; trivial one-offs decay fast:

```
stability = base_days · (0.5 + importance) · (1 + ln(1 + access_count))
retention = 0.5 ^ (days_since_access / stability)
```

A memory is forgotten when `retention < forget_threshold` **and** it is not
protected (`importance ≥ protect_importance`, e.g. preferences and identity).
Forgetting is **reversible-by-audit**: dropped memories are appended to
`memory_archive.jsonl` with their retention at time of forgetting, not deleted
into the void.

**Reinforcement** is the counterweight: every time a memory is recalled and
used, its `last_accessed` resets and `access_count` increments — raising both
its recency and its stability. Memory that keeps proving useful sticks.

## Consolidation (no duplicate bloat)

On write, a new memory is compared against its nearest neighbour:

- **cosine ≥ `dedup_threshold` (0.88):** auto-reinforce the existing memory
  (taking the higher importance) instead of storing a redundant copy.
- **cosine in `[dedup_gray_low, dedup_threshold)` (0.70–0.88):** ask an **LLM
  judge** "are these the same fact?". If yes, merge into one *canonical*
  phrasing (the more complete/specific version) and reinforce. This catches
  paraphrases that pure cosine misses — e.g. *"I like Python"* vs *"the user's
  favourite language is Python"*.
- **otherwise:** store as new.

Known limitation: online dedup is pairwise against the single nearest
neighbour, so it is order-dependent — two paraphrases can survive as separate
clusters if a third memory sits between them. A periodic full-pass
consolidation (O(n²) judge calls, run rarely) would mop up residual duplicates;
`Consolidator.judge_merge` is the building block.

## Memory summarization (compress, don't just drop)

`agent.compress_memories()` gathers unprotected `fact`/`event` memories older
than `compress_min_age_days` and, if there are at least `compress_min_cluster`
of them, asks the LLM to distil them into ≤2 durable notes. The originals are
archived and replaced by the notes (`source="consolidation"`). This keeps the
long tail of aged, individually-low-value memories from bloating the store
while preserving any signal worth carrying forward — a softer alternative to
outright forgetting.

## Context packing (recall within a limited window)

Re-ranking gives an ordering; packing decides what actually fits. We use
**Maximal Marginal Relevance** over the re-ranked list, bounded by
`context_token_budget`:

```
MMR = λ · score(m) − (1 − λ) · max cosine(m, already_selected)
```

This trades relevance against diversity (`λ = mmr_lambda`), so the injected
block doesn't spend a tight token budget on five near-identical memories.
Selected memories are grouped by type (`profile / preference / instruction /
fact / event`) into the system prompt.

## Cross-session persistence

Chroma persists to `MEMORYAGENT_STORE` (default `~/.memoryagent`). A brand-new
`MemoryAgent` process pointed at the same path recalls everything the previous
sessions learned — demonstrated in `demo.py` (session 2 is a fresh agent that
recalls session 1's preferences).

## Layout

```
memoryagent/
  config.py            # all tunables: weights, thresholds, budgets
  llm.py               # Qwen client + offline fallback (embedder & stub LLM)
  agent.py             # retrieve → respond → reflect orchestration
  cli.py               # interactive REPL (/mem /why /forget /stats /clear)
  memory/
    schema.py          # Memory record + Chroma (de)serialization
    scoring.py         # composite re-rank + Ebbinghaus forgetting  ◀ core
    store.py           # Chroma store: recall, reinforce, consolidate, sweep
    extractor.py       # LLM reflection → atomic memories
    consolidation.py   # LLM-judged dedup-merge + summarization
    packer.py          # token-budgeted MMR selection + formatting
demo.py                # cross-session / forgetting / reinforcement demo
demo_consolidation.py  # dedup-merge + summarization demo
eval_retrieval.py      # precision@k / recall@k / MRR — composite vs semantic
```

## Evaluation

`eval_retrieval.py` seeds a labelled memory set, runs queries, and reports
`precision@k`, `recall@k`, and `MRR`, comparing the composite re-ranker against
a pure-semantic baseline. On real Qwen embeddings (12 memories, 7 queries,
k=3), composite lifts recall@3 from 0.93 → **1.00** and precision@3 from
0.38 → **0.43** by letting importance surface a second relevant memory the
semantic score alone ranked too low. The tradeoff is a small MRR dip
(0.93 → 0.86): importance can occasionally demote an exact-but-trivial match
from first place. (P@3 tops out near 0.33 for single-relevant queries — a
metric artifact of k=3, not a ranking failure.)

## Tuning cheat-sheet

| Want | Knob (`config.py`) |
|------|--------------------|
| More weight on semantic match | `RetrievalWeights.semantic` |
| Memory to fade faster | ↓ `recency_half_life_days`, ↓ `retention_base_days` |
| Forget more aggressively | ↑ `forget_threshold` |
| Protect more as permanent | ↓ `protect_importance` |
| Less redundant recall | ↓ `mmr_lambda` |
| Bigger recalled context | ↑ `context_token_budget`, ↑ `return_k` |
| Stricter auto-dedup | ↑ `dedup_threshold` |
| Send more pairs to the LLM judge | ↓ `dedup_gray_low` |
| Compress memories sooner / in smaller clusters | ↓ `compress_min_age_days`, ↓ `compress_min_cluster` |

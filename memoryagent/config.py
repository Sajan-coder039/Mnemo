"""Central configuration for MemoryAgent.

Everything tunable about retrieval ranking and forgetting lives here so the
behaviour can be tuned without touching the algorithms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Load a .env file into the process environment (dependency-free).
# Runs at import — before the config defaults below read os.environ — so every
# entry point (CLI, demos, eval) picks up keys without a manual `export`.
# Real environment variables take precedence over .env.
# --------------------------------------------------------------------------- #
def _load_dotenv() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    seen = set()
    # Search the package dir, the project root, and the current directory.
    for base in (here, os.path.dirname(here), os.getcwd()):
        path = os.path.join(base, ".env")
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)  # don't override a real env var


_load_dotenv()


# --------------------------------------------------------------------------- #
# Qwen Cloud (DashScope, OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
# International endpoint. For the mainland-China endpoint use:
#   https://dashscope.aliyuncs.com/compatible-mode/v1
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMConfig:
    api_key: str = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url: str = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
    chat_model: str = os.environ.get("QWEN_CHAT_MODEL", "qwen-plus")
    embed_model: str = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v3")
    embed_dim: int = int(os.environ.get("QWEN_EMBED_DIM", "1024"))
    temperature: float = 0.4

    @property
    def offline(self) -> bool:
        """No API key → run with the local embedder + stub LLM."""
        return not self.api_key


@dataclass
class RetrievalWeights:
    """Weights for the composite re-ranking score (need not sum to 1)."""
    semantic: float = 0.55
    recency: float = 0.20
    importance: float = 0.15
    frequency: float = 0.10


@dataclass
class MemoryConfig:
    # Where Chroma + the archive live (persists across sessions/processes).
    store_path: str = os.environ.get(
        "MEMORYAGENT_STORE", os.path.expanduser("~/.memoryagent")
    )

    weights: RetrievalWeights = None  # set in __post_init__

    # Two-stage retrieval
    recall_k: int = 24          # ANN candidates pulled from Chroma
    return_k: int = 8           # memories after re-rank + packing

    # Recency scoring: score halves every `recency_half_life_days`.
    recency_half_life_days: float = 7.0
    # Frequency saturation constant (higher = slower saturation).
    frequency_k: float = 5.0

    # Forgetting (Ebbinghaus retention curve)
    retention_base_days: float = 3.0    # base memory stability
    forget_threshold: float = 0.15      # retention below this → forget
    protect_importance: float = 0.75    # importance at/above this is never forgotten

    # Consolidation
    dedup_threshold: float = 0.88       # cosine ≥ this → auto-merge (reinforce)
    dedup_gray_low: float = 0.70        # cosine in [gray_low, threshold) → ask the LLM judge
    compress_min_age_days: float = 7.0  # only compress memories older than this
    compress_min_cluster: int = 3       # need at least this many to bother compressing

    # Context packing
    context_token_budget: int = 800     # tokens reserved for recalled memories
    mmr_lambda: float = 0.7             # relevance vs. diversity in MMR (1=all relevance)

    # Conversation window kept verbatim in-context (older turns live in memory).
    history_turns: int = 6

    def __post_init__(self):
        if self.weights is None:
            self.weights = RetrievalWeights()


DEFAULT_LLM = LLMConfig()
DEFAULT_MEMORY = MemoryConfig()

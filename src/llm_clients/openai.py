import os
from functools import lru_cache
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Single source of truth for all production model choices.
# Change a tier here and every call site picks it up.
MODEL_CONFIG = {
    "router":         {"model": "gpt-4o-mini", "temperature": 0},
    "generator":      {"model": "gpt-4o-mini", "temperature": 0},
    "query_rewriter": {"model": "gpt-4o-mini", "temperature": 0.1},
    "embedding":      {"model": "text-embedding-3-large"},
}


def _api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found.")
    return key


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Cached sync OpenAI client. Built once per process."""
    return OpenAI(api_key=_api_key())


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    """Cached async OpenAI client. Built once per process."""
    return AsyncOpenAI(api_key=_api_key())


@lru_cache(maxsize=None)
def get_chat_model(task: str) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI for a named production task.
    Task names are keys of MODEL_CONFIG.
    Cached per task so each ChatOpenAI is built once per process.
    """
    if task not in MODEL_CONFIG:
        raise KeyError(f"Unknown task '{task}'. Known tasks: {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[task]
    return ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])


@lru_cache(maxsize=1)
def get_chroma_embedding_function():
    """Cached ChromaDB-compatible embedding function. Built once per process."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=_api_key(),
        model_name=MODEL_CONFIG["embedding"]["model"],
    )


# --- Backward-compat exports for src/evaluation/ scripts ---
# Resolved lazily via __getattr__ so importing this module costs nothing
# until a client is actually requested.
def __getattr__(name):
    if name == "client":
        return get_openai_client()
    if name == "async_client":
        return get_async_openai_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

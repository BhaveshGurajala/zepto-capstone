"""Settings, all read from environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = "zepto_policies"

# Local open-source embedding model. EMBEDDING_MODEL may also be a local folder path.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 3

# Optional real-LLM settings (only used when MOCK_LLM=0)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_EXTRA_RETRIES = 2  # retries after the first attempt if the LLM's JSON fails validation


def mock_llm() -> bool:
    """True unless MOCK_LLM is explicitly set to "0".

    Unset or "1" -> deterministic mock mode (the graded default, no network).
    Read on every call so tests can flip it without restarting.
    """
    return os.getenv("MOCK_LLM", "1").strip() != "0"

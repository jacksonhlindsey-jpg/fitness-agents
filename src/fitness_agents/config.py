"""Settings loaded from the environment (and .env, if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _key(name: str, default: str = "") -> str:
    """An API key from the environment, without stray spaces or quotes from copy-pasting."""
    return os.getenv(name, default).strip().strip("'\"").strip()


GEMINI_API_KEY = _key("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# gemini (free key), ollama (local, no key), or anthropic (paid). See llm.py.
PROVIDER = os.getenv("FITNESS_PROVIDER") or ("gemini" if GEMINI_API_KEY else "ollama")
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}
MODEL = os.getenv("FITNESS_MODEL") or DEFAULT_MODELS[PROVIDER]
# Quick jobs (routing a message, parsing a food log) use a faster model on Gemini;
# the planners and the Reviewer use MODEL.
FAST_MODEL = os.getenv("FITNESS_FAST_MODEL") or ("gemini-3.5-flash-lite" if PROVIDER == "gemini" else MODEL)

USDA_API_KEY = _key("USDA_API_KEY") or "DEMO_KEY"

# Storage: SQLite file locally; Postgres when DATABASE_URL is set (hosted).
DB_PATH = Path(os.getenv("FITNESS_DB", "data/fitness.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Hosted: every browser gets its own data (identified by an X-User-Id header).
# Local: the CLI and the web UI share one profile.
MULTI_USER = os.getenv("FITNESS_MULTI_USER", "") == "1"

# How many times the Reviewer may send a plan back before it goes to the user anyway.
MAX_REVISIONS = 2

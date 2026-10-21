"""Settings loaded from the environment (and .env, if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# gemini (free key), ollama (local, no key), or anthropic (paid). See llm.py.
PROVIDER = os.getenv("FITNESS_PROVIDER") or ("gemini" if GEMINI_API_KEY else "ollama")
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}
MODEL = os.getenv("FITNESS_MODEL") or DEFAULT_MODELS[PROVIDER]

USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
DB_PATH = Path(os.getenv("FITNESS_DB", "data/fitness.db"))

# How many times the Reviewer may send a plan back before it goes to the user anyway.
MAX_REVISIONS = 2

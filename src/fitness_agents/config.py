"""Settings loaded from the environment (and .env, if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("FITNESS_MODEL", "claude-opus-5-5")
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
DB_PATH = Path(os.getenv("FITNESS_DB", "data/fitness.db"))

# How many times the Reviewer may send a plan back before it goes to the user anyway.
MAX_REVISIONS = 2

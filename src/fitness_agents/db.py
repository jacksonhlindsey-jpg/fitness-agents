"""Local SQLite storage: the user profile, daily logs, approved plans, and a food lookup cache.

History lives here rather than in the LLM context, so agents only get the
summaries they need (see the long-context risk in the proposal).
"""

import json
import sqlite3
from datetime import date, datetime, timedelta

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS food_log (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,
    food TEXT NOT NULL,
    grams REAL,
    calories REAL NOT NULL,
    protein_g REAL NOT NULL,
    carbs_g REAL NOT NULL,
    fat_g REAL NOT NULL,
    fiber_g REAL,
    source TEXT NOT NULL            -- 'usda' or 'estimate'
);
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,
    kind TEXT NOT NULL,             -- run, walk, lift, cardio, other
    description TEXT NOT NULL,
    distance_mi REAL,
    duration_min REAL,
    exercise TEXT,
    sets INTEGER,
    reps INTEGER,
    weight_lb REAL
);
CREATE TABLE IF NOT EXISTS metric_log (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,
    metric TEXT NOT NULL,           -- body_weight_lb, water_oz
    value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,             -- meal or workout
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS food_cache (
    query TEXT PRIMARY KEY,
    data TEXT                       -- per-100g nutrients as JSON, or NULL if not found
);
"""


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def today() -> str:
    return date.today().isoformat()


# --- profile -----------------------------------------------------------------

def get_profile() -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT data FROM profile WHERE id = 1").fetchone()
    return json.loads(row["data"]) if row else None


def save_profile(profile: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO profile (id, data) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (json.dumps(profile),),
        )


# --- logs --------------------------------------------------------------------

def log_food(entry: dict, day: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO food_log (day, food, grams, calories, protein_g, carbs_g, fat_g, fiber_g, source)"
            " VALUES (:day, :food, :grams, :calories, :protein_g, :carbs_g, :fat_g, :fiber_g, :source)",
            {"day": day or today(), "fiber_g": None, **entry},
        )


def log_activity(entry: dict, day: str | None = None) -> None:
    row = {
        "day": day or today(),
        "distance_mi": None, "duration_min": None, "exercise": None,
        "sets": None, "reps": None, "weight_lb": None,
        **entry,
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO activity_log (day, kind, description, distance_mi, duration_min, exercise, sets, reps, weight_lb)"
            " VALUES (:day, :kind, :description, :distance_mi, :duration_min, :exercise, :sets, :reps, :weight_lb)",
            row,
        )


def log_metric(metric: str, value: float, day: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO metric_log (day, metric, value) VALUES (?, ?, ?)", (day or today(), metric, value)
        )


def day_totals(day: str | None = None) -> dict:
    """Calories, macros, and water logged on one day."""
    day = day or today()
    with connect() as conn:
        food = conn.execute(
            "SELECT COALESCE(SUM(calories),0) calories, COALESCE(SUM(protein_g),0) protein_g,"
            " COALESCE(SUM(carbs_g),0) carbs_g, COALESCE(SUM(fat_g),0) fat_g,"
            " COALESCE(SUM(fiber_g),0) fiber_g FROM food_log WHERE day = ?",
            (day,),
        ).fetchone()
        water = conn.execute(
            "SELECT COALESCE(SUM(value),0) FROM metric_log WHERE day = ? AND metric = 'water_oz'", (day,)
        ).fetchone()[0]
    return {**{k: round(food[k]) for k in food.keys()}, "water_oz": round(water)}


def history(days: int = 30) -> dict:
    """Recent raw history for the progress summary."""
    since = (date.today() - timedelta(days=days)).isoformat()
    with connect() as conn:
        weights = conn.execute(
            "SELECT day, value FROM metric_log WHERE metric = 'body_weight_lb' AND day >= ? ORDER BY day, id",
            (since,),
        ).fetchall()
        calories = conn.execute(
            "SELECT day, SUM(calories) total FROM food_log WHERE day >= ? GROUP BY day ORDER BY day", (since,)
        ).fetchall()
        runs = conn.execute(
            "SELECT day, distance_mi, duration_min FROM activity_log"
            " WHERE kind = 'run' AND day >= ? ORDER BY day",
            (since,),
        ).fetchall()
        lifts = conn.execute(
            "SELECT day, exercise, sets, reps, weight_lb FROM activity_log"
            " WHERE kind = 'lift' AND day >= ? ORDER BY day",
            (since,),
        ).fetchall()
    return {
        "weights": [dict(r) for r in weights],
        "calories": [dict(r) for r in calories],
        "runs": [dict(r) for r in runs],
        "lifts": [dict(r) for r in lifts],
    }


# --- plans -------------------------------------------------------------------

def save_plan(kind: str, plan: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO plans (created_at, kind, data) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), kind, json.dumps(plan)),
        )


def latest_plan(kind: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT data FROM plans WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()
    return json.loads(row["data"]) if row else None


# --- food cache --------------------------------------------------------------

_MISSING = object()


def cached_food(query: str):
    """Return cached nutrients, None if cached as not-found, or _MISSING if never looked up."""
    with connect() as conn:
        row = conn.execute("SELECT data FROM food_cache WHERE query = ?", (query,)).fetchone()
    if row is None:
        return _MISSING
    return json.loads(row["data"]) if row["data"] else None


def cache_food(query: str, nutrients: dict | None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO food_cache (query, data) VALUES (?, ?)",
            (query, json.dumps(nutrients) if nutrients else None),
        )

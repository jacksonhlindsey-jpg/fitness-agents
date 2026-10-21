"""Storage: user profiles, daily logs, approved plans, and a food lookup cache.

SQLite on your computer by default; Postgres when DATABASE_URL is set (the hosted
version, where the server's own disk is wiped on every restart).

Every row belongs to a user_id, so each visitor to the hosted site only ever sees
their own data. The CLI and the local web UI share LOCAL_USER.

History lives here rather than in the LLM context, so agents only get the
summaries they need (see the long-context risk in the proposal).
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from . import config

LOCAL_USER = "local"

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS food_log (
    id {id},
    user_id TEXT NOT NULL,
    day TEXT NOT NULL,
    food TEXT NOT NULL,
    grams {real},
    calories {real} NOT NULL,
    protein_g {real} NOT NULL,
    carbs_g {real} NOT NULL,
    fat_g {real} NOT NULL,
    fiber_g {real},
    source TEXT NOT NULL            -- 'usda' or 'estimate'
);
CREATE TABLE IF NOT EXISTS activity_log (
    id {id},
    user_id TEXT NOT NULL,
    day TEXT NOT NULL,
    kind TEXT NOT NULL,             -- run, walk, lift, cardio, other
    description TEXT NOT NULL,
    distance_mi {real},
    duration_min {real},
    exercise TEXT,
    sets INTEGER,
    reps INTEGER,
    weight_lb {real}
);
CREATE TABLE IF NOT EXISTS metric_log (
    id {id},
    user_id TEXT NOT NULL,
    day TEXT NOT NULL,
    metric TEXT NOT NULL,           -- body_weight_lb, water_oz
    value {real} NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id {id},
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,             -- meal or workout
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS food_cache (
    query TEXT PRIMARY KEY,
    data TEXT                       -- per-100g nutrients as JSON, or NULL if not found
);
CREATE INDEX IF NOT EXISTS food_log_user_day ON food_log (user_id, day);
CREATE INDEX IF NOT EXISTS activity_log_user_day ON activity_log (user_id, day);
CREATE INDEX IF NOT EXISTS metric_log_user_day ON metric_log (user_id, day);
CREATE INDEX IF NOT EXISTS plans_user_kind ON plans (user_id, kind);
"""

TYPES = {
    "sqlite": {"id": "INTEGER PRIMARY KEY", "real": "REAL"},
    "postgres": {"id": "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY", "real": "DOUBLE PRECISION"},
}

_ready: set[str] = set()  # databases whose schema has been checked this process


class _Conn:
    """The few query helpers this module needs, the same for SQLite and Postgres.

    SQL is written with `?` placeholders and translated for Postgres.
    """

    def __init__(self, conn, postgres: bool):
        self.conn, self.postgres = conn, postgres

    def run(self, sql: str, params: tuple = ()):
        if self.postgres:
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def all(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.run(sql, params).fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        row = self.run(sql, params).fetchone()
        return dict(row) if row else None


@contextmanager
def _db():
    """A connection that commits on success. Creates or migrates the schema on first use."""
    if config.DATABASE_URL:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(config.DATABASE_URL, row_factory=dict_row) as conn:
            if config.DATABASE_URL not in _ready:
                for statement in _schema("postgres"):
                    conn.execute(statement)
                _ready.add(config.DATABASE_URL)
            yield _Conn(conn, postgres=True)
    else:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            if str(config.DB_PATH) not in _ready:
                _migrate_single_user_sqlite(conn)
                conn.executescript(";\n".join(_schema("sqlite")))
                _ready.add(str(config.DB_PATH))
            with conn:
                yield _Conn(conn, postgres=False)
        finally:
            conn.close()


def _schema(backend: str) -> list[str]:
    sql = SCHEMA.format(**TYPES[backend])
    lines = [line.split("--")[0] for line in sql.splitlines()]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _migrate_single_user_sqlite(conn: sqlite3.Connection) -> None:
    """Upgrade a database from before per-user data: existing rows become LOCAL_USER's."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    with conn:
        if "profile" in tables:
            conn.execute("CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO profiles (user_id, data) SELECT ?, data FROM profile", (LOCAL_USER,))
            conn.execute("DROP TABLE profile")
        for table in ("food_log", "activity_log", "metric_log", "plans"):
            if table not in tables:
                continue
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if "user_id" not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT '{LOCAL_USER}'")


def today() -> str:
    return date.today().isoformat()


# --- profile -----------------------------------------------------------------

def get_profile(user_id: str) -> dict | None:
    with _db() as db:
        row = db.one("SELECT data FROM profiles WHERE user_id = ?", (user_id,))
    return json.loads(row["data"]) if row else None


def save_profile(user_id: str, profile: dict) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO profiles (user_id, data) VALUES (?, ?)"
            " ON CONFLICT (user_id) DO UPDATE SET data = excluded.data",
            (user_id, json.dumps(profile)),
        )


# --- logs --------------------------------------------------------------------

def log_food(user_id: str, entry: dict, day: str | None = None) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO food_log (user_id, day, food, grams, calories, protein_g, carbs_g, fat_g, fiber_g, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, day or today(), entry["food"], entry.get("grams"), entry["calories"], entry["protein_g"],
             entry["carbs_g"], entry["fat_g"], entry.get("fiber_g"), entry["source"]),
        )


def log_activity(user_id: str, entry: dict, day: str | None = None) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO activity_log"
            " (user_id, day, kind, description, distance_mi, duration_min, exercise, sets, reps, weight_lb)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, day or today(), entry["kind"], entry["description"], entry.get("distance_mi"),
             entry.get("duration_min"), entry.get("exercise"), entry.get("sets"), entry.get("reps"),
             entry.get("weight_lb")),
        )


def log_metric(user_id: str, metric: str, value: float, day: str | None = None) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO metric_log (user_id, day, metric, value) VALUES (?, ?, ?, ?)",
            (user_id, day or today(), metric, value),
        )


def day_totals(user_id: str, day: str | None = None) -> dict:
    """Calories, macros, and water logged on one day."""
    day = day or today()
    with _db() as db:
        food = db.one(
            "SELECT COALESCE(SUM(calories), 0) AS calories, COALESCE(SUM(protein_g), 0) AS protein_g,"
            " COALESCE(SUM(carbs_g), 0) AS carbs_g, COALESCE(SUM(fat_g), 0) AS fat_g,"
            " COALESCE(SUM(fiber_g), 0) AS fiber_g FROM food_log WHERE user_id = ? AND day = ?",
            (user_id, day),
        )
        water = db.one(
            "SELECT COALESCE(SUM(value), 0) AS water_oz FROM metric_log"
            " WHERE user_id = ? AND day = ? AND metric = 'water_oz'",
            (user_id, day),
        )
    return {k: round(v) for k, v in {**food, **water}.items()}


def day_entries(user_id: str, day: str | None = None) -> dict:
    """Everything logged on one day, for the Today view."""
    day = day or today()
    with _db() as db:
        foods = db.all(
            "SELECT food, grams, calories, protein_g, source FROM food_log"
            " WHERE user_id = ? AND day = ? ORDER BY id",
            (user_id, day),
        )
        activities = db.all(
            "SELECT kind, description, distance_mi, duration_min, exercise, sets, reps, weight_lb"
            " FROM activity_log WHERE user_id = ? AND day = ? ORDER BY id",
            (user_id, day),
        )
    return {"foods": foods, "activities": activities}


def history(user_id: str, days: int = 30) -> dict:
    """Recent raw history for the progress summary."""
    since = (date.today() - timedelta(days=days)).isoformat()
    with _db() as db:
        return {
            "weights": db.all(
                "SELECT day, value FROM metric_log"
                " WHERE user_id = ? AND metric = 'body_weight_lb' AND day >= ? ORDER BY day, id",
                (user_id, since),
            ),
            "calories": db.all(
                "SELECT day, SUM(calories) AS total FROM food_log"
                " WHERE user_id = ? AND day >= ? GROUP BY day ORDER BY day",
                (user_id, since),
            ),
            "runs": db.all(
                "SELECT day, distance_mi, duration_min FROM activity_log"
                " WHERE user_id = ? AND kind = 'run' AND day >= ? ORDER BY day, id",
                (user_id, since),
            ),
            "lifts": db.all(
                "SELECT day, exercise, sets, reps, weight_lb FROM activity_log"
                " WHERE user_id = ? AND kind = 'lift' AND day >= ? ORDER BY day, id",
                (user_id, since),
            ),
        }


# --- plans -------------------------------------------------------------------

def save_plan(user_id: str, kind: str, plan: dict) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO plans (user_id, created_at, kind, data) VALUES (?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(timespec="seconds"), kind, json.dumps(plan)),
        )


def latest_plan(user_id: str, kind: str) -> dict | None:
    with _db() as db:
        row = db.one(
            "SELECT data FROM plans WHERE user_id = ? AND kind = ? ORDER BY id DESC LIMIT 1", (user_id, kind)
        )
    return json.loads(row["data"]) if row else None


# --- food cache (shared by all users: it's public USDA data) ------------------

_MISSING = object()


def cached_food(query: str):
    """Return cached nutrients, None if cached as not-found, or _MISSING if never looked up."""
    with _db() as db:
        row = db.one("SELECT data FROM food_cache WHERE query = ?", (query,))
    if row is None:
        return _MISSING
    return json.loads(row["data"]) if row["data"] else None


def cache_food(query: str, nutrients: dict | None) -> None:
    with _db() as db:
        db.run(
            "INSERT INTO food_cache (query, data) VALUES (?, ?)"
            " ON CONFLICT (query) DO UPDATE SET data = excluded.data",
            (query, json.dumps(nutrients) if nutrients else None),
        )

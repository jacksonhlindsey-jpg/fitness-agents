"""Upgrading a database created before per-user data."""

import json
import sqlite3

from fitness_agents import config, db


def test_old_single_user_database_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE profile (id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL);
        CREATE TABLE food_log (id INTEGER PRIMARY KEY, day TEXT NOT NULL, food TEXT NOT NULL, grams REAL,
            calories REAL NOT NULL, protein_g REAL NOT NULL, carbs_g REAL NOT NULL, fat_g REAL NOT NULL,
            fiber_g REAL, source TEXT NOT NULL);
        CREATE TABLE plans (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
    """)
    old.execute("INSERT INTO profile VALUES (1, ?)", (json.dumps({"age": 22}),))
    old.execute("INSERT INTO food_log (day, food, calories, protein_g, carbs_g, fat_g, source)"
                " VALUES ('2026-10-21', 'egg', 70, 6, 0, 5, 'usda')")
    old.execute("INSERT INTO plans (created_at, kind, data) VALUES ('2026-10-21', 'meal', '{\"x\": 1}')")
    old.commit()
    old.close()

    monkeypatch.setattr(config, "DB_PATH", path)
    assert db.get_profile(db.LOCAL_USER) == {"age": 22}
    assert db.day_totals(db.LOCAL_USER, "2026-10-21")["calories"] == 70
    assert db.latest_plan(db.LOCAL_USER, "meal") == {"x": 1}
    assert db.get_profile("someone-else") is None

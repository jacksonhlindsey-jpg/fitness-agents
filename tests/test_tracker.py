"""Tracker with a fake LLM and a fake USDA lookup."""

import pytest

from fitness_agents import config, db, llm, nutrition_api
from fitness_agents.db import LOCAL_USER as ME
from fitness_agents.agents.tracker import Activity, FoodItem, ParsedLog, tracker

PROFILE = {"height_in": 70, "weight_lb": 200, "goal_weight_lb": 180, "age": 21, "sex": "male",
           "activity": "moderate", "weekly_rate_lb": 1.0, "max_lifts": {"bench": 200}}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.save_profile(ME, dict(PROFILE))


def test_logs_food_from_usda_and_updates_max(monkeypatch):
    parsed = ParsedLog(
        foods=[
            FoodItem(name="2 eggs", search_term="egg, scrambled", grams=100, est_calories=150,
                     est_protein_g=0, est_carbs_g=0, est_fat_g=0),
            FoodItem(name="mystery bar", search_term="zzz", grams=50, est_calories=200,
                     est_protein_g=10, est_carbs_g=20, est_fat_g=8),
        ],
        activities=[Activity(kind="lift", description="bench 205x5", distance_mi=None, duration_min=None,
                             exercise="bench", sets=1, reps=5, weight_lb=205)],
        water_oz=16, body_weight_lb=None, needs_clarification=False, clarification_question="",
    )
    monkeypatch.setattr(llm, "ask", lambda *a, **k: parsed)
    usda = {"description": "Egg, scrambled", "calories": 149, "protein_g": 10, "carbs_g": 1.6, "fat_g": 11, "fiber_g": 0}
    monkeypatch.setattr(nutrition_api, "per_100g", lambda q: usda if q.startswith("egg") else None)

    out = tracker({"user_id": ME, "user_input": "...", "profile": db.get_profile(ME)})["response"]

    assert db.day_totals(ME)["calories"] == 349  # 149 from USDA + 200 estimated
    assert "(estimated)" in out
    assert db.get_profile(ME)["max_lifts"]["bench"] == 239.2


def test_ambiguous_input_asks_instead_of_logging(monkeypatch):
    parsed = ParsedLog(foods=[], activities=[], water_oz=None, body_weight_lb=None,
                       needs_clarification=True,
                       clarification_question="Roughly how much pasta, about one cup or a full plate?")
    monkeypatch.setattr(llm, "ask", lambda *a, **k: parsed)
    out = tracker({"user_id": ME, "user_input": "had some pasta", "profile": PROFILE})["response"]
    assert "pasta" in out
    assert db.day_totals(ME)["calories"] == 0


def test_usda_ranking_prefers_plain_food():
    foods = [{"description": d} for d in
             ("Bananas, dehydrated, or banana powder", "Banana chips", "Banana, raw", "Bananas, raw")]
    assert nutrition_api._best_match("banana, raw", foods)["description"] in ("Banana, raw", "Bananas, raw")


def test_bad_usda_match_falls_back_to_estimate(monkeypatch):
    parsed = ParsedLog(
        foods=[FoodItem(name="banana", search_term="banana, raw", grams=120, est_calories=105,
                        est_protein_g=1, est_carbs_g=27, est_fat_g=0)],
        activities=[], water_oz=None, body_weight_lb=None, needs_clarification=False, clarification_question="",
    )
    monkeypatch.setattr(llm, "ask", lambda *a, **k: parsed)
    dried = {"description": "Bananas, dehydrated", "calories": 346, "protein_g": 3.9, "carbs_g": 88, "fat_g": 1.8, "fiber_g": 10}
    monkeypatch.setattr(nutrition_api, "per_100g", lambda q: dried)
    tracker({"user_id": ME, "user_input": "a banana", "profile": PROFILE})
    assert db.day_totals(ME)["calories"] == 105

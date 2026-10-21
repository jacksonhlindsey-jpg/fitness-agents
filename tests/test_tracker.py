"""The Tracker agent loop with a scripted fake LLM and a fake USDA lookup (no network)."""

import pytest

from fitness_agents import config, db, llm, nutrition_api
from fitness_agents.agents.tracker import (
    MAX_STEPS, AskUser, Finish, GetTodayTotals, LogActivity, LogBodyMetric, LogFood, SearchFood, Step, tracker,
)
from fitness_agents.db import LOCAL_USER as ME

PROFILE = {"height_in": 70, "weight_lb": 200, "goal_weight_lb": 180, "age": 21, "sex": "male",
           "activity": "moderate", "weekly_rate_lb": 1.0, "max_lifts": {"bench": 200}}
EGG = {"description": "Egg, whole, cooked, scrambled", "calories": 149, "protein_g": 10, "carbs_g": 1.6, "fat_g": 11, "fiber_g": 0}
DRIED_BANANA = {"description": "Bananas, dehydrated", "calories": 346, "protein_g": 3.9, "carbs_g": 88, "fat_g": 1.8, "fiber_g": 10}
BANANA = {"description": "Banana, raw", "calories": 89, "protein_g": 1.1, "carbs_g": 22.8, "fat_g": 0.3, "fiber_g": 2.6}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.save_profile(ME, dict(PROFILE))


def script(monkeypatch, *steps):
    """Make the LLM return these steps in order; returns the list of prompts it received."""
    prompts, queue = [], list(steps)
    monkeypatch.setattr(llm, "ask", lambda system, prompt, schema, effort="low": (prompts.append(prompt), queue.pop(0))[1])
    return prompts


def usda(monkeypatch, table):
    monkeypatch.setattr(nutrition_api, "search", lambda q: table[q])


def log_food(name, grams, match, kcal):
    return LogFood(tool="log_food", name=name, grams=grams, usda_match=match,
                   est_calories=kcal, est_protein_g=5, est_carbs_g=5, est_fat_g=5)


def run(text="..."):
    return tracker({"user_id": ME, "user_input": text, "profile": db.get_profile(ME)})


def test_full_run_uses_tools_and_records_a_trace(monkeypatch):
    usda(monkeypatch, {"egg, scrambled": [EGG], "banana, raw": [DRIED_BANANA, BANANA]})
    prompts = script(
        monkeypatch,
        Step(thought="Search both foods first.", actions=[
            SearchFood(tool="search_food", query="egg, scrambled"),
            SearchFood(tool="search_food", query="banana, raw")]),
        Step(thought="Log them with the fresh banana, the lift, and water.", actions=[
            log_food("2 eggs", 100, "Egg, whole, cooked, scrambled", 150),
            log_food("banana", 118, "Banana, raw", 105),
            LogActivity(tool="log_activity", kind="lift", description="bench 205x5", distance_mi=None,
                        duration_min=None, exercise="bench", sets=1, reps=5, weight_lb=205),
            LogBodyMetric(tool="log_body_metric", metric="water_oz", value=16),
            GetTodayTotals(tool="get_today_totals")]),
        Step(thought="Done.", actions=[Finish(tool="finish", answer="Add a protein-rich lunch.")]),
    )
    out = run("had 2 eggs and a banana, bench 205x5, 2 cups water")

    assert db.day_totals(ME)["calories"] == 149 + 105  # 105 = 89 kcal/100 g * 118 g
    assert db.day_totals(ME)["water_oz"] == 16
    assert db.get_profile(ME)["max_lifts"]["bench"] == 239.2
    assert out["response"].endswith("Add a protein-rich lunch.")
    assert [s["step"] for s in out["trace"]] == [1, 2, 3, "answer"]
    assert out["trace"][1]["calls"][4]["observation"]["remaining"]["calories"] > 0
    # Each step sees the earlier observations.
    assert "Banana, raw" in prompts[1] and "Result:" in prompts[2]


def test_vague_input_ends_with_a_question(monkeypatch):
    script(monkeypatch, Step(thought="Too vague.", actions=[AskUser(tool="ask_user", question="How much pasta?")]))
    out = run("had some pasta")
    assert out["response"] == "How much pasta?"
    assert out["trace"][-1] == {"step": "answer", "type": "ask_user", "text": "How much pasta?"}
    assert db.day_totals(ME)["calories"] == 0


def test_step_limit_stops_a_looping_agent(monkeypatch):
    usda(monkeypatch, {"egg": [EGG]})
    looping = Step(thought="Search again.", actions=[SearchFood(tool="search_food", query="egg")])
    prompts = script(monkeypatch, *[looping] * 10)
    out = run("egg")
    assert len(prompts) == MAX_STEPS
    assert out["trace"][-1]["type"] == "step_limit"
    assert "stopped after" in out["response"]


def test_bad_match_is_rejected_and_the_agent_recovers(monkeypatch):
    usda(monkeypatch, {"banana, raw": [BANANA]})
    script(
        monkeypatch,
        Step(thought="Search.", actions=[SearchFood(tool="search_food", query="banana, raw")]),
        Step(thought="Log.", actions=[log_food("banana", 118, "Banana, ripe and lovely", 105)]),
        Step(thought="Fix the match.", actions=[log_food("banana", 118, "Banana, raw", 105)]),
        Step(thought="Done.", actions=[Finish(tool="finish", answer="")]),
    )
    out = run("a banana")
    assert "error" in out["trace"][1]["calls"][0]["observation"]
    assert db.day_totals(ME)["calories"] == 105


def test_usda_down_falls_back_to_the_estimate(monkeypatch):
    def down(q):
        raise nutrition_api.LookupFailed("USDA could not be reached")
    monkeypatch.setattr(nutrition_api, "search", down)
    script(
        monkeypatch,
        Step(thought="Search.", actions=[SearchFood(tool="search_food", query="banana, raw")]),
        Step(thought="Use my estimate.", actions=[log_food("banana", 118, None, 105)]),
        Step(thought="Done.", actions=[Finish(tool="finish", answer="")]),
    )
    out = run("a banana")
    assert "could not be reached" in out["trace"][0]["calls"][0]["observation"]["error"]
    assert "(estimated)" in out["response"]
    assert db.day_totals(ME)["calories"] == 105


def test_wrong_food_guardrail_and_no_double_logging(monkeypatch):
    usda(monkeypatch, {"banana": [DRIED_BANANA]})
    script(
        monkeypatch,
        Step(thought="Search.", actions=[SearchFood(tool="search_food", query="banana")]),
        Step(thought="Log twice by mistake.", actions=[
            log_food("banana", 118, "Bananas, dehydrated", 105),
            log_food("banana", 118, "Bananas, dehydrated", 105)]),
        Step(thought="Done.", actions=[Finish(tool="finish", answer="")]),
    )
    out = run("a banana")
    first, second = out["trace"][1]["calls"]
    assert "estimate was used" in first["observation"]["note"]  # 408 kcal from USDA vs 105 estimated
    assert "already logged" in second["observation"]["error"]
    assert db.day_totals(ME)["calories"] == 105


def test_usda_ranking_prefers_plain_food():
    foods = [{"description": d} for d in
             ("Bananas, dehydrated, or banana powder", "Banana chips", "Banana, raw", "Bananas, raw")]
    assert nutrition_api.rank("banana, raw", foods)[0]["description"] in ("Banana, raw", "Bananas, raw")

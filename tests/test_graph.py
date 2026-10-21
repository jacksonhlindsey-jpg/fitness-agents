"""Exercise the whole workflow with a fake LLM: no API key or cost needed."""

import uuid

import pytest
from langgraph.types import Command

from fitness_agents import config, db, llm
from fitness_agents.db import LOCAL_USER
from fitness_agents.agents.nutrition_planner import Meal, MealItem, MealPlan
from fitness_agents.agents.orchestrator import Routing
from fitness_agents.agents.reviewer import Critique
from fitness_agents.agents.workout_planner import Exercise, WorkoutDay, WorkoutPlan
from fitness_agents.graph import build_graph

PROFILE = {
    "height_in": 70, "weight_lb": 200, "goal_weight_lb": 180, "age": 21, "sex": "male",
    "activity": "moderate", "weekly_rate_lb": 1.0, "food_likes": "chicken, rice, eggs",
    "food_avoid": "peanuts", "training_days": 3, "equipment": "full gym", "avoid_exercises": "",
    "max_lifts": {"bench": 225, "squat": 205, "deadlift": 315, "ohp": 135},
}


def meal_plan(kcal_per_meal: float) -> MealPlan:
    item = MealItem(food="chicken and rice", amount="1 bowl", calories=kcal_per_meal,
                    protein_g=50, carbs_g=80, fat_g=15)
    return MealPlan(meals=[Meal(name=f"Meal {i}", items=[item], swap_options=["eggs"]) for i in range(3)], notes="")


WORKOUT = WorkoutPlan(
    days=[WorkoutDay(day=d, focus="full body", exercises=[
        Exercise(name="Back squat", sets=4, reps="5", weight_lb=160, muscles=["quads"]),
        Exercise(name="Bench press", sets=3, reps="5", weight_lb=170, muscles=["chest"]),
    ]) for d in ("Mon", "Wed", "Fri")],
    weak_point_strategy="Extra squat volume.", notes="",
)


@pytest.fixture
def fake_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.save_profile(LOCAL_USER, PROFILE)
    calls: list[str] = []
    meal_drafts = iter([meal_plan(300), meal_plan(820)])  # first draft is far under target

    def ask(system, prompt, schema, effort="low"):
        calls.append(schema.__name__)
        if schema is Routing:
            return Routing(intent="full_plan", reply="")
        if schema is MealPlan:
            return next(meal_drafts)
        if schema is WorkoutPlan:
            return WORKOUT
        if schema is Critique:
            return Critique(meal_issues=[], workout_issues=[])
        raise AssertionError(schema)

    monkeypatch.setattr(llm, "ask", ask)
    return calls


def test_full_plan_runs_planners_in_parallel_revises_and_waits_for_approval(fake_llm):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"user_id": LOCAL_USER, "user_input": "make me a full plan"}, cfg)

    # Both planners ran, the Reviewer's calorie rule rejected the first meal plan,
    # and only the Nutrition Planner was asked to redo its work.
    assert fake_llm[0] == "Routing"
    assert sorted(fake_llm[1:3]) == ["MealPlan", "WorkoutPlan"]  # parallel, so either order
    assert fake_llm[3:] == ["Critique", "MealPlan", "Critique"]
    assert "__interrupt__" in result
    assert result["meal_plan"]["totals"]["calories"] == 2460

    # Nothing is saved until the user approves.
    assert db.latest_plan(LOCAL_USER, "meal") is None
    result = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert result["approved"]
    assert db.latest_plan(LOCAL_USER, "meal")["totals"]["calories"] == 2460
    assert db.latest_plan(LOCAL_USER, "workout") is not None

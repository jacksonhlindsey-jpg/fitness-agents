"""Web API with a fake LLM: profile, chat, and the approval round trip."""

import pytest
from fastapi.testclient import TestClient

from fitness_agents import config, db, llm, server
from fitness_agents.agents.orchestrator import Routing
from fitness_agents.agents.reviewer import Critique
from tests.test_graph import WORKOUT, meal_plan

PROFILE = {
    "height_in": 68, "weight_lb": 140, "goal_weight_lb": 150, "age": 22, "sex": "male",
    "activity": "moderate", "weekly_rate_lb": 1, "food_likes": "eggs, beef", "training_days": 4,
    "max_lifts": {"bench": "165", "squat": "185x5", "deadlift": "", "ohp": ""},
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")

    def ask(system, prompt, schema, effort="low"):
        if schema is Routing:
            return Routing(intent="meal_plan", reply="")
        if schema is Critique:
            return Critique(meal_issues=[], workout_issues=[])
        return meal_plan(1000) if schema.__name__ == "MealPlan" else WORKOUT

    monkeypatch.setattr(llm, "ask", ask)
    return TestClient(server.app)


def test_profile_round_trip(client):
    assert client.get("/api/profile").json()["profile"] is None
    saved = client.put("/api/profile", json=PROFILE).json()
    assert saved["profile"]["max_lifts"] == {"bench": 165.0, "squat": 215.8}
    assert saved["summary"]["calories"]["target"] > saved["summary"]["calories"]["maintenance"]
    assert client.get("/api/profile").json()["profile"]["age"] == 22


def test_bad_lift_is_rejected(client):
    bad = {**PROFILE, "max_lifts": {"bench": "a lot"}}
    assert client.put("/api/profile", json=bad).status_code == 422


def test_plan_waits_for_approval_then_saves(client):
    client.put("/api/profile", json=PROFILE)
    pending = client.post("/api/chat", json={"message": "make me a meal plan"}).json()
    assert pending["type"] == "approval" and pending["meal_plan"]
    assert client.get("/api/plans").json()["meal_plan"] is None

    done = client.post("/api/approve", json={"thread_id": pending["thread_id"], "action": "approve"}).json()
    assert done["type"] == "message"
    assert client.get("/api/plans").json()["meal_plan"]["totals"]["calories"] == 3000

    # The same plan can't be approved twice.
    again = client.post("/api/approve", json={"thread_id": pending["thread_id"], "action": "approve"})
    assert again.status_code == 404


def test_today_and_progress(client):
    client.put("/api/profile", json=PROFILE)
    db.log_metric("body_weight_lb", 141)
    today = client.get("/api/today").json()
    assert today["summary"]["water_goal_oz"] == 70
    progress = client.get("/api/progress").json()
    assert progress["current_weight_lb"] == 141

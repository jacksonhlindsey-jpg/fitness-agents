"""Workout Planner: builds a weekly schedule that targets the user's weakest lifts."""

import json

from pydantic import BaseModel

from .. import calc, llm
from ..state import FitnessState
from .nutrition_planner import _feedback

SYSTEM = """You are the Workout Planner in a fitness app. Build a one-week training schedule.

Rules:
- Use exactly the number of training days the user has available; every other day is rest.
- Give extra volume and priority to the listed weak points, without neglecting the other lifts.
- Base working weights on the user's max lifts (e.g. 70-80% of 1RM for strength work).
- Only use equipment the user has. Never program exercises on their avoid list; pick an alternative.
- Stay within about 10-20 hard sets per muscle group per week.
- If the user is trying to lose weight, include some cardio and keep volume moderate.
If reviewer or user feedback is included, fix every point it raises."""


class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    weight_lb: float | None
    muscles: list[str]


class WorkoutDay(BaseModel):
    day: str
    focus: str
    exercises: list[Exercise]


class WorkoutPlan(BaseModel):
    days: list[WorkoutDay]
    weak_point_strategy: str
    notes: str


def workout_planner(state: FitnessState) -> FitnessState:
    profile = state["profile"]
    strength = calc.strength_profile(profile.get("max_lifts", {}), profile["weight_lb"], profile["sex"])

    prompt = {
        "training_days_per_week": profile.get("training_days", 3),
        "max_lifts_lb": profile.get("max_lifts", {}),
        "strength_scores": strength["scores"],
        "weak_points": strength["weak_points"],
        "equipment": profile.get("equipment", "full gym"),
        "avoid_exercises": profile.get("avoid_exercises", ""),
        "goal": f"{profile['weight_lb']} lb -> {profile['goal_weight_lb']} lb",
    }
    feedback = _feedback(state, "workout")
    if feedback:
        prompt["previous_plan"] = state.get("workout_plan")
        prompt["feedback_to_fix"] = feedback

    plan = llm.ask(SYSTEM, json.dumps(prompt, indent=2), WorkoutPlan, effort="medium")
    result = plan.model_dump()
    result["weak_points"] = strength["weak_points"]
    return {"workout_plan": result}

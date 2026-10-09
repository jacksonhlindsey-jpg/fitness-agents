"""The shared state passed between agents in the LangGraph workflow."""

from typing import Literal, TypedDict

Intent = Literal["log", "meal_plan", "workout_plan", "full_plan", "progress", "chat"]
Planner = Literal["nutrition_planner", "workout_planner"]


class FitnessState(TypedDict, total=False):
    user_input: str
    profile: dict
    intent: Intent
    planners: list[Planner]     # which planners this request needs

    meal_plan: dict | None
    workout_plan: dict | None
    review: dict                # {"meal": [issues], "workout": [issues]}
    revisions: int              # Reviewer send-backs so far
    user_feedback: str          # edit request from the human approval step
    approved: bool

    response: str               # final text shown to the user

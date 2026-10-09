"""Nutrition Planner: builds a one-day meal plan around the user's calorie target and food preferences."""

import json

from pydantic import BaseModel

from .. import calc, llm
from ..state import FitnessState

SYSTEM = """You are the Nutrition Planner in a fitness app. Build a realistic one-day meal plan.

Rules:
- Hit the calorie target within about 5% and get close to the protein target.
- Build meals from foods the user likes. Never include anything on their avoid list.
- Use ordinary portions with amounts a person can measure (grams, cups, pieces).
- Give each meal 1-2 swap options in case the user doesn't want it.
- Keep it simple: 3-5 meals or snacks.
If reviewer or user feedback is included, fix every point it raises."""


class MealItem(BaseModel):
    food: str
    amount: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class Meal(BaseModel):
    name: str
    items: list[MealItem]
    swap_options: list[str]


class MealPlan(BaseModel):
    meals: list[Meal]
    notes: str


def nutrition_planner(state: FitnessState) -> FitnessState:
    profile = state["profile"]
    target = calc.calorie_target(profile)
    macros = calc.macro_targets(target["target"], profile["weight_lb"])

    prompt = {
        "calorie_target": target["target"],
        "macro_targets": macros,
        "goal": f"{profile['weight_lb']} lb -> {profile['goal_weight_lb']} lb",
        "foods_they_like": profile.get("food_likes", ""),
        "avoid": profile.get("food_avoid", ""),
    }
    feedback = _feedback(state, "meal")
    if feedback:
        prompt["previous_plan"] = state.get("meal_plan")
        prompt["feedback_to_fix"] = feedback

    plan = llm.ask(SYSTEM, json.dumps(prompt, indent=2), MealPlan, effort="medium")
    result = plan.model_dump()
    # Recompute totals in code rather than trusting the model's arithmetic.
    result["totals"] = {
        key: round(sum(item[key] for meal in result["meals"] for item in meal["items"]))
        for key in ("calories", "protein_g", "carbs_g", "fat_g")
    }
    result["calorie_target"] = target["target"]
    result["macro_targets"] = macros
    return {"meal_plan": result}


def _feedback(state: FitnessState, kind: str) -> list[str]:
    """Reviewer issues for this plan kind, plus any edit request from the user."""
    items = list((state.get("review") or {}).get(kind, []))
    if state.get("user_feedback"):
        items.append(f"User request: {state['user_feedback']}")
    return items

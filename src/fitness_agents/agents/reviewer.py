"""Reviewer (critic): checks every plan before the user sees it.

Two layers:
1. Hard rules in code (calorie floor, calorie accuracy, avoided foods/exercises,
   rest days, weekly volume). These can't be talked around by the model.
2. An LLM critique for things rules can't catch (does the plan actually target
   the weak points, are the portions realistic, does it match the user's goal).
"""

import json
import re
from collections import Counter

from pydantic import BaseModel

from .. import calc, llm
from ..state import FitnessState

SYSTEM = """You are the Reviewer in a fitness app. You check plans written by other agents before the user sees them.

Look for: unsafe advice, plans that don't fit the user's goal, workouts that don't actually
prioritize the listed weak points, unrealistic portions or meals, and anything that ignores
the user's preferences. Only report real problems, each as one short, specific, fixable
sentence. Return empty lists if a plan is fine. Don't repeat issues listed under
`already_found`."""

MAX_WEEKLY_SETS_PER_MUSCLE = 22
MAX_SETS_PER_SESSION = 30


class Critique(BaseModel):
    meal_issues: list[str]
    workout_issues: list[str]


def reviewer(state: FitnessState) -> FitnessState:
    profile = state["profile"]
    meal, workout = state.get("meal_plan"), state.get("workout_plan")

    issues = {
        "meal": check_meal_plan(meal, profile) if meal else [],
        "workout": check_workout_plan(workout, profile) if workout else [],
    }

    prompt = {
        "user": {k: profile.get(k) for k in ("weight_lb", "goal_weight_lb", "sex", "age", "food_likes",
                                             "food_avoid", "avoid_exercises", "equipment")},
        "meal_plan": meal,
        "workout_plan": workout,
        "already_found": issues,
    }
    critique = llm.ask(SYSTEM, json.dumps(prompt, indent=2), Critique, effort="medium")
    if meal:
        issues["meal"] += critique.meal_issues
    if workout:
        issues["workout"] += critique.workout_issues

    return {"review": issues, "revisions": state.get("revisions", 0) + 1}


def check_meal_plan(plan: dict, profile: dict) -> list[str]:
    issues = []
    total = plan["totals"]["calories"]
    target = plan["calorie_target"]
    floor = calc.CALORIE_FLOOR[profile["sex"]]
    if total < floor:
        issues.append(f"Plan totals {total} kcal, below the safe minimum of {floor} kcal.")
    if abs(total - target) > 0.1 * target:
        issues.append(f"Plan totals {total} kcal but the target is {target} kcal; adjust portions.")
    for food in _avoided(plan_text=json.dumps(plan["meals"]), avoid=profile.get("food_avoid", "")):
        issues.append(f"Plan includes '{food}', which the user wants to avoid.")
    return issues


def check_workout_plan(plan: dict, profile: dict) -> list[str]:
    issues = []
    training_days = [d for d in plan["days"] if d["exercises"]]
    available = profile.get("training_days", 3)
    if len(training_days) > available:
        issues.append(f"Plan has {len(training_days)} training days but the user has only {available}.")
    if len(training_days) >= 7:
        issues.append("Plan has no rest day.")

    weekly_sets: Counter[str] = Counter()
    for day in training_days:
        session = sum(ex["sets"] for ex in day["exercises"])
        if session > MAX_SETS_PER_SESSION:
            issues.append(f"{day['day']} has {session} sets; keep sessions under {MAX_SETS_PER_SESSION}.")
        for ex in day["exercises"]:
            for muscle in ex["muscles"]:
                weekly_sets[muscle.lower()] += ex["sets"]
    for muscle, sets in weekly_sets.items():
        if sets > MAX_WEEKLY_SETS_PER_MUSCLE:
            issues.append(f"{sets} weekly sets for {muscle} is too much; stay under {MAX_WEEKLY_SETS_PER_MUSCLE}.")

    names = " ".join(ex["name"] for d in training_days for ex in d["exercises"])
    for exercise in _avoided(plan_text=names, avoid=profile.get("avoid_exercises", "")):
        issues.append(f"Plan includes '{exercise}', which the user wants to avoid.")
    return issues


def _avoided(plan_text: str, avoid: str) -> list[str]:
    """Items from a comma-separated avoid list that appear in the plan text."""
    text = plan_text.lower()
    items = [a.strip().lower() for a in re.split(r"[,;]", avoid) if a.strip()]
    return [item for item in items if re.search(rf"\b{re.escape(item)}", text)]

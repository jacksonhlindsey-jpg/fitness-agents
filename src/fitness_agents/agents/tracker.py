"""Tracker: turns free-text entries ("two eggs and toast, ran 2 miles") into structured logs.

Calories come from USDA FoodData Central when a match is found; the LLM's own
estimate is only a fallback, and every entry records which source was used.
"""

from typing import Literal

from pydantic import BaseModel

from .. import calc, db, llm, nutrition_api
from ..state import FitnessState

SYSTEM = """You are the Tracker agent in a fitness app. Convert the user's message into structured log entries.

Foods: give a short generic USDA search term that includes the preparation
(e.g. "egg, scrambled", "bread, whole wheat, toasted", "banana, raw", "rice, white, cooked")
and your best estimate of the portion in grams. Also estimate calories and macros for that
portion; these are only used if the database has no match.

Activities: runs/walks need distance in miles and/or duration; lifts need the exercise name
using one of: bench, squat, deadlift, ohp, or another plain name, plus sets, reps, and weight in lb.

Only if something essential is too ambiguous to log (e.g. "some pasta" with no hint of amount),
set needs_clarification to true, put one short question in clarification_question, and leave
the lists empty. Otherwise set needs_clarification to false and leave the question empty.
Assume normal portions when a reasonable default exists ("a bowl of cereal", "a banana")."""


class FoodItem(BaseModel):
    name: str
    search_term: str
    grams: float
    est_calories: float
    est_protein_g: float
    est_carbs_g: float
    est_fat_g: float


class Activity(BaseModel):
    kind: Literal["run", "walk", "lift", "cardio", "other"]
    description: str
    distance_mi: float | None
    duration_min: float | None
    exercise: str | None
    sets: int | None
    reps: int | None
    weight_lb: float | None


class ParsedLog(BaseModel):
    foods: list[FoodItem]
    activities: list[Activity]
    water_oz: float | None
    body_weight_lb: float | None
    needs_clarification: bool
    clarification_question: str


def tracker(state: FitnessState) -> FitnessState:
    parsed = llm.ask(SYSTEM, state["user_input"], ParsedLog, effort="low")
    if parsed.needs_clarification and parsed.clarification_question:
        return {"response": parsed.clarification_question}

    user_id = state["user_id"]
    lines: list[str] = []
    for item in parsed.foods:
        lines.append(_log_food(user_id, item))
    for activity in parsed.activities:
        db.log_activity(user_id, activity.model_dump())
        lines.append(f"Logged {activity.description}")
    if parsed.water_oz:
        db.log_metric(user_id, "water_oz", parsed.water_oz)
        lines.append(f"Logged {parsed.water_oz:g} oz water")
    if parsed.body_weight_lb:
        db.log_metric(user_id, "body_weight_lb", parsed.body_weight_lb)
        lines.append(f"Logged weight {parsed.body_weight_lb:g} lb")

    profile = state.get("profile") or {}
    lines += _update_maxes(user_id, profile, parsed.activities, parsed.body_weight_lb)
    if parsed.foods:
        lines += _daily_summary(user_id, profile)

    return {"response": "\n".join(lines) or "I didn't find anything to log in that message."}


def _log_food(user_id: str, item: FoodItem) -> str:
    per_100g = nutrition_api.per_100g(item.search_term)
    values = nutrition_api.scale(per_100g, item.grams) if per_100g else None
    # Guard against a bad database match: if USDA and the estimate disagree by more
    # than 2x, the match is probably the wrong food (e.g. dried instead of fresh).
    if values and item.est_calories and not 0.5 <= values["calories"] / item.est_calories <= 2:
        values = None
    if values:
        source = "usda"
    else:
        values = {
            "calories": item.est_calories, "protein_g": item.est_protein_g,
            "carbs_g": item.est_carbs_g, "fat_g": item.est_fat_g, "fiber_g": None,
        }
        source = "estimate"
    db.log_food(user_id, {"food": item.name, "grams": item.grams, "source": source, **values})
    tag = "" if source == "usda" else " (estimated)"
    return f"Logged {item.name} (~{item.grams:g} g): {values['calories']:.0f} kcal{tag}"


def _update_maxes(user_id: str, profile: dict, activities: list[Activity], new_weight: float | None) -> list[str]:
    """Raise stored max lifts when a logged set implies a higher one-rep max."""
    if not profile:
        return []
    notes = []
    maxes = profile.setdefault("max_lifts", {})
    for a in activities:
        if a.kind != "lift" or a.exercise not in calc.STRENGTH_STANDARDS["male"] or not (a.weight_lb and a.reps):
            continue
        estimate = calc.epley_1rm(a.weight_lb, a.reps)
        if estimate > (maxes.get(a.exercise) or 0):
            maxes[a.exercise] = estimate
            notes.append(f"New estimated {a.exercise} max: {estimate:g} lb")
    if new_weight:
        profile["weight_lb"] = new_weight
    if notes or new_weight:
        db.save_profile(user_id, profile)
    return notes


def _daily_summary(user_id: str, profile: dict) -> list[str]:
    totals = db.day_totals(user_id)
    if not profile:
        return [f"Today so far: {totals['calories']} kcal"]
    target = calc.calorie_target(profile)["target"]
    macros = calc.macro_targets(target, profile["weight_lb"])
    lines = [
        f"\nToday: {totals['calories']} / {target} kcal"
        f" | protein {totals['protein_g']}/{macros['protein_g']} g"
        f" | carbs {totals['carbs_g']}/{macros['carbs_g']} g"
        f" | fat {totals['fat_g']}/{macros['fat_g']} g"
    ]
    if totals["calories"] > target * 1.1:
        lines.append("You're over today's target.")
    protein_left = macros["protein_g"] - totals["protein_g"]
    if protein_left > 30:
        lines.append(f"You still need about {protein_left} g of protein. "
                     "Easy options: Greek yogurt, chicken breast, eggs, cottage cheese, or a protein shake.")
    return lines

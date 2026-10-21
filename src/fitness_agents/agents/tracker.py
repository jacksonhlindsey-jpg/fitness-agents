"""Tracker agent: a tool-using loop that turns "two eggs and a banana, ran 2 miles" into logged data.

Each step, the model writes a short thought and one or more tool calls. The tools
run, their results come back as observations, and the model takes another step,
until it calls `finish` or `ask_user`, or hits MAX_STEPS (the stop condition).
Every step is recorded as a trace: thought -> tool calls -> observations -> answer.

The model decides what to look up and which USDA result matches what the user
ate; the math stays in code (log_food scales the chosen match's per-100 g values).
"""

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .. import calc, db, llm, nutrition_api
from ..state import FitnessState

MAX_STEPS = 6

SYSTEM = """You are the Tracker agent in a fitness app. You log what the user ate, drank, weighed,
and did by calling tools. Work in steps: each step, write a short thought, then one or more
tool calls. You get every tool's result back before your next step.

Tools:
- search_food(query): look up a food in the USDA database. Use a short generic term that includes
  the preparation, e.g. "egg, scrambled", "banana, raw", "rice, white, cooked". Returns up to 5
  candidates with calories and macros per 100 g. Search every food before you log it.
- log_food(name, grams, usda_match, est_calories, est_protein_g, est_carbs_g, est_fat_g): save one
  food. name is what the user ate ("2 scrambled eggs"); grams is the portion. usda_match is the exact
  description of the search result that is the same food (prefer the plain version over dried,
  powdered, or restaurant versions unless the user said so), or null if none fits or search failed.
  The app calculates calories from usda_match; your est_* numbers for the whole portion are only
  used when usda_match is null.
- log_activity(kind, description, distance_mi, duration_min, exercise, sets, reps, weight_lb): runs
  and walks need distance and/or duration; lifts need exercise (bench, squat, deadlift, ohp, or a
  plain name), sets, reps, and weight in lb.
- log_body_metric(metric, value): body_weight_lb, or water_oz (convert cups, bottles, liters to oz).
- get_today_totals(): today's calories and macros against the user's targets. Call it after logging food.
- ask_user(question): only if something essential is too vague to estimate (e.g. "some pasta").
  Ends your turn.
- finish(answer): ends your turn. answer is one or two short sentences of coaching based on today's
  totals, such as foods that would close a protein gap. Don't repeat the numbers; the app shows them.

Log each item exactly once. Assume normal portions when a reasonable default exists ("a banana",
"a bowl of cereal"). A typical run: step 1 search all foods; step 2 log everything and get today's
totals; step 3 finish."""


# --- tool calls the model can make ---------------------------------------------

class SearchFood(BaseModel):
    tool: Literal["search_food"]
    query: str


class LogFood(BaseModel):
    tool: Literal["log_food"]
    name: str
    grams: float
    usda_match: str | None
    est_calories: float
    est_protein_g: float
    est_carbs_g: float
    est_fat_g: float


class LogActivity(BaseModel):
    tool: Literal["log_activity"]
    kind: Literal["run", "walk", "lift", "cardio", "other"]
    description: str
    distance_mi: float | None
    duration_min: float | None
    exercise: str | None
    sets: int | None
    reps: int | None
    weight_lb: float | None


class LogBodyMetric(BaseModel):
    tool: Literal["log_body_metric"]
    metric: Literal["body_weight_lb", "water_oz"]
    value: float


class GetTodayTotals(BaseModel):
    tool: Literal["get_today_totals"]


class AskUser(BaseModel):
    tool: Literal["ask_user"]
    question: str


class Finish(BaseModel):
    tool: Literal["finish"]
    answer: str


Action = Annotated[
    Union[SearchFood, LogFood, LogActivity, LogBodyMetric, GetTodayTotals, AskUser, Finish],
    Field(discriminator="tool"),
]


class Step(BaseModel):
    thought: str
    actions: list[Action]


# --- the agent loop ------------------------------------------------------------

def tracker(state: FitnessState) -> FitnessState:
    session = _Session(state["user_id"], state.get("profile") or {})
    transcript = f"User message: {state['user_input']}"
    trace: list[dict] = []
    outcome: dict | None = None

    for number in range(1, MAX_STEPS + 1):
        step = llm.ask(SYSTEM, transcript + "\n\nTake your next step.", Step, effort="low")
        calls = []
        for action in step.actions:
            if isinstance(action, (Finish, AskUser)):
                text = action.answer if isinstance(action, Finish) else action.question
                outcome = {"type": action.tool, "text": text}
                break
            observation = session.run(action)
            calls.append({"tool": action.tool, "input": action.model_dump(exclude={"tool"}), "observation": observation})
        if not step.actions:
            calls.append({"tool": None, "input": {}, "observation": {"error": "Call at least one tool."}})

        trace.append({"step": number, "thought": step.thought, "calls": calls,
                      "model": getattr(llm.last_call, "model", None), "seconds": getattr(llm.last_call, "seconds", None)})
        transcript += _render_step(number, step.thought, calls)
        if outcome:
            break
    else:
        outcome = {"type": "step_limit", "text": f"I stopped after {MAX_STEPS} steps without finishing."}

    trace.append({"step": "answer", **outcome})
    return {"response": session.response(outcome), "trace": trace}


def _render_step(number: int, thought: str, calls: list[dict]) -> str:
    lines = [f"\n\nStep {number}", f"Thought: {thought}"]
    for call in calls:
        lines.append(f"Call: {call['tool']} {json.dumps(call['input'])}")
        lines.append(f"Result: {json.dumps(call['observation'])}")
    return "\n".join(lines)


class _Session:
    """Runs tool calls for one Tracker turn and remembers what it did."""

    def __init__(self, user_id: str, profile: dict):
        self.user_id, self.profile = user_id, profile
        self.candidates: dict[str, dict] = {}  # USDA description (lowercase) -> per-100 g nutrients
        self.logged: list[str] = []             # lines for the reply
        self.logged_keys: set[tuple] = set()
        self.logged_food = False

    def run(self, action) -> dict | list:
        handler = getattr(self, f"_{action.tool}")
        return handler(action)

    # tools

    def _search_food(self, a: SearchFood):
        try:
            results = nutrition_api.search(a.query)
        except nutrition_api.LookupFailed as e:
            return {"error": f"{e}. Log this food with usda_match null and your own estimate."}
        if not results:
            return {"result": "No matches. Try a simpler term, or log with usda_match null and your estimate."}
        for r in results:
            self.candidates[r["description"].lower()] = r
        return [{"match": r["description"], "per_100g": {k: r[k] for k in ("calories", "protein_g", "carbs_g", "fat_g")}}
                for r in results]

    def _log_food(self, a: LogFood):
        key = ("food", a.name.lower(), round(a.grams))
        if key in self.logged_keys:
            return {"error": f"{a.name} is already logged."}

        estimate = {"calories": a.est_calories, "protein_g": a.est_protein_g,
                    "carbs_g": a.est_carbs_g, "fat_g": a.est_fat_g, "fiber_g": None}
        note = None
        if a.usda_match:
            match = self.candidates.get(a.usda_match.lower())
            if match is None:
                return {"error": "usda_match must be the exact description of a search result, or null."}
            values, source = nutrition_api.scale(match, a.grams), "usda"
            # Guardrail: a USDA value more than 2x off the estimate usually means the wrong food.
            if a.est_calories and not 0.5 <= values["calories"] / a.est_calories <= 2:
                values, source = estimate, "estimate"
                note = "USDA match was far from your estimate, so your estimate was used."
        else:
            values, source = estimate, "estimate"

        db.log_food(self.user_id, {"food": a.name, "grams": a.grams, "source": source, **values})
        self.logged_keys.add(key)
        self.logged_food = True
        tag = "" if source == "usda" else " (estimated)"
        self.logged.append(f"Logged {a.name} (~{a.grams:g} g): {values['calories']:.0f} kcal{tag}")
        result = {"logged": a.name, "grams": a.grams, "calories": values["calories"],
                  "protein_g": values["protein_g"], "source": f"USDA: {match['description']}" if source == "usda" else "estimate"}
        return {**result, "note": note} if note else result

    def _log_activity(self, a: LogActivity):
        key = ("activity", a.description.lower())
        if key in self.logged_keys:
            return {"error": f"{a.description} is already logged."}
        db.log_activity(self.user_id, a.model_dump(exclude={"tool"}))
        self.logged_keys.add(key)
        self.logged.append(f"Logged {a.description}")
        result = {"logged": a.description}
        if a.kind == "lift" and a.exercise in calc.STRENGTH_STANDARDS["male"] and a.weight_lb and a.reps:
            estimate = calc.epley_1rm(a.weight_lb, a.reps)
            maxes = self.profile.setdefault("max_lifts", {})
            if self.profile and estimate > (maxes.get(a.exercise) or 0):
                maxes[a.exercise] = estimate
                db.save_profile(self.user_id, self.profile)
                self.logged.append(f"New estimated {a.exercise} max: {estimate:g} lb")
                result["new_estimated_max_lb"] = estimate
        return result

    def _log_body_metric(self, a: LogBodyMetric):
        key = ("metric", a.metric)
        if key in self.logged_keys:
            return {"error": f"{a.metric} is already logged."}
        db.log_metric(self.user_id, a.metric, a.value)
        self.logged_keys.add(key)
        if a.metric == "body_weight_lb":
            self.logged.append(f"Logged weight {a.value:g} lb")
            if self.profile:
                self.profile["weight_lb"] = a.value
                db.save_profile(self.user_id, self.profile)
        else:
            self.logged.append(f"Logged {a.value:g} oz water")
        return {"logged": a.metric, "value": a.value}

    def _get_today_totals(self, a: GetTodayTotals):
        totals = db.day_totals(self.user_id)
        if not self.profile:
            return {"eaten": totals, "targets": "unknown (no profile yet)"}
        targets = self._targets()
        remaining = {k: round(targets[k] - totals[k]) for k in targets}
        return {"eaten": totals, "targets": targets, "remaining": remaining}

    # reply

    def _targets(self) -> dict:
        calories = calc.calorie_target(self.profile)["target"]
        return {"calories": calories, **calc.macro_targets(calories, self.profile["weight_lb"])}

    def response(self, outcome: dict) -> str:
        lines = list(self.logged)
        if self.logged_food:
            totals = db.day_totals(self.user_id)
            if self.profile:
                t = self._targets()
                lines.append(f"\nToday: {totals['calories']} / {t['calories']} kcal"
                             f" | protein {totals['protein_g']}/{t['protein_g']} g"
                             f" | carbs {totals['carbs_g']}/{t['carbs_g']} g"
                             f" | fat {totals['fat_g']}/{t['fat_g']} g")
            else:
                lines.append(f"\nToday so far: {totals['calories']} kcal")
        if outcome["text"]:
            lines.append(("\n" if lines else "") + outcome["text"])
        return "\n".join(lines) or "I didn't find anything to log in that message."

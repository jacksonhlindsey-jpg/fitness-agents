"""Command-line chat for the fitness agents.

Run with:  uv run fitness-agents
"""

import re
import uuid

from langgraph.types import Command

from . import calc, config, db, llm
from .display import format_meal_plan, format_trace, format_workout_plan
from .graph import build_graph

DISCLAIMER = (
    "Note: this is a class project, not medical advice. If you have a medical condition\n"
    "or a history of disordered eating, talk to a doctor or dietitian first."
)
HELP = """Just type naturally, for example:
  had 2 eggs and toast, ran 2 miles in 18 min, drank 20 oz water
  bench 185 for 5
  make me a meal plan  /  build me a workout plan  /  make me a full plan
  how am I doing?
Commands: /profile (set up or redo profile), /today, /trace (how the agent handled your last log), /help, /quit"""


def main() -> None:
    print(f"Fitness Agents  (model: {config.MODEL} via {config.PROVIDER})\n" + DISCLAIMER + "\n")
    if not db.get_profile(db.LOCAL_USER):
        print("Let's set up your profile first.\n")
        onboard()
    print(HELP)

    graph = build_graph()
    last_trace = None
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return
        if text == "/help":
            print(HELP)
        elif text == "/profile":
            onboard()
        elif text == "/trace":
            print(format_trace(last_trace) if last_trace else "No agent trace yet. Log something first.")
        elif text == "/today":
            t = db.day_totals(db.LOCAL_USER)
            print(f"Today: {t['calories']} kcal | protein {t['protein_g']} g | carbs {t['carbs_g']} g"
                  f" | fat {t['fat_g']} g | water {t['water_oz']} oz")
        else:
            last_trace = run_turn(graph, text) or last_trace


def run_turn(graph, text: str) -> list[dict] | None:
    """Run one message through the agents. Returns the Tracker's trace, if it ran."""
    # Each message gets its own thread, so plan state doesn't leak between requests.
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        result = graph.invoke({"user_id": db.LOCAL_USER, "user_input": text}, cfg)
        while "__interrupt__" in result:
            decision = ask_approval(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=decision), cfg)
    except llm.LLMError as e:
        print(f"agent> Sorry, something went wrong: {e}")
        return None
    print(f"agent> {result.get('response', '')}")
    if result.get("trace"):
        steps = sum(1 for s in result["trace"] if s["step"] != "answer")
        print(f"       (Tracker agent: {steps} step{'s' if steps != 1 else ''}; type /trace to see them)")
    return result.get("trace")


def ask_approval(payload: dict) -> dict:
    """Human-in-the-loop checkpoint: nothing is saved until the user approves it."""
    if payload["meal_plan"]:
        print("\n" + format_meal_plan(payload["meal_plan"]))
    if payload["workout_plan"]:
        print("\n" + format_workout_plan(payload["workout_plan"]))
    unresolved = [i for issues in payload["unresolved_issues"].values() for i in issues]
    if unresolved:
        print("\nThe Reviewer still flagged:\n  - " + "\n  - ".join(unresolved))

    while True:
        choice = input("\n[a]pprove, [e]dit, or [r]eject? ").strip().lower()
        if choice.startswith("a"):
            return {"action": "approve"}
        if choice.startswith("r"):
            return {"action": "reject"}
        if choice.startswith("e"):
            targets = [k for k in ("meal", "workout") if payload[f"{k}_plan"]]
            if len(targets) > 1:
                which = input("Edit which? [m]eal, [w]orkout, [b]oth: ").strip().lower()
                targets = {"m": ["meal"], "w": ["workout"]}.get(which[:1], targets)
            feedback = input("What should change? ").strip()
            return {"action": "edit", "targets": targets, "feedback": feedback}


# --- onboarding ---------------------------------------------------------------

def onboard() -> None:
    old = db.get_profile(db.LOCAL_USER) or {}
    p: dict = {}
    feet, inches = _ask("Height (e.g. 5'10)", _parse_height, old.get("height_in"))
    p["height_in"] = feet * 12 + inches
    p["weight_lb"] = _ask("Current weight (lb)", float, old.get("weight_lb"))
    p["goal_weight_lb"] = _ask("Goal weight (lb)", float, old.get("goal_weight_lb"))
    p["age"] = _ask("Age", int, old.get("age"))
    p["sex"] = _ask("Sex for calorie formulas (male/female)", _choice("male", "female"), old.get("sex"))
    p["activity"] = _ask(
        "Activity level (sedentary/light/moderate/active/very_active)",
        _choice(*calc.ACTIVITY_MULTIPLIERS), old.get("activity"),
    )
    if p["goal_weight_lb"] != p["weight_lb"]:
        p["weekly_rate_lb"] = _ask("How many lb per week to change (e.g. 1)", float, old.get("weekly_rate_lb", 1.0))
    p["food_likes"] = _ask("Foods you like (comma separated)", str, old.get("food_likes", ""))
    p["food_avoid"] = _ask("Foods to avoid or allergies (comma separated, or blank)", str, old.get("food_avoid", ""), allow_blank=True)
    p["training_days"] = _ask("Training days per week", int, old.get("training_days", 3))
    p["equipment"] = _ask("Equipment (e.g. full gym, dumbbells only)", str, old.get("equipment", "full gym"))
    p["avoid_exercises"] = _ask("Exercises to avoid (comma separated, or blank)", str, old.get("avoid_exercises", ""), allow_blank=True)

    print("\nMax lifts: enter a 1-rep max (225) or a recent set (185x5), or leave blank.")
    old_maxes = old.get("max_lifts", {})
    p["max_lifts"] = {
        lift: _ask(f"  {lift}", calc.parse_lift, old_maxes.get(lift), allow_blank=True)
        for lift in ("bench", "squat", "deadlift", "ohp")
    }

    db.save_profile(db.LOCAL_USER, p)
    target = calc.calorie_target(p)
    value = calc.bmi(p["weight_lb"], p["height_in"])
    print(f"\nSaved. BMI {value} ({calc.bmi_category(value)}). "
          f"Maintenance ~{target['maintenance']} kcal, daily target {target['target']} kcal"
          f" ({target['weekly_rate_lb']:+g} lb/week).")
    if target["floor_applied"]:
        print("Your target was raised to the safe minimum, so progress will be a bit slower than requested.")


def _ask(prompt, parse, default=None, allow_blank=False):
    hint = f" [{default}]" if default not in (None, "") else ""
    while True:
        raw = input(f"{prompt}{hint}: ").strip()
        if not raw:
            if default not in (None, ""):
                return _reparse(parse, default)
            if allow_blank:
                return "" if parse is str else None
            continue
        try:
            return parse(raw)
        except ValueError as e:
            print(f"  {e or 'Please try again.'}")


def _reparse(parse, default):
    # The stored height is total inches, but the parser returns (feet, inches).
    if parse is _parse_height:
        return divmod(int(default), 12)
    return default


def _parse_height(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d)\s*['f]?\s*(\d{1,2})?\s*(?:\"|in)?", raw.replace("ft", "'"))
    if not m:
        raise ValueError("Use a format like 5'10")
    return int(m.group(1)), int(m.group(2) or 0)


def _choice(*options):
    def parse(raw: str) -> str:
        raw = raw.lower()
        if raw not in options:
            raise ValueError(f"Choose one of: {', '.join(options)}")
        return raw
    return parse


if __name__ == "__main__":
    main()

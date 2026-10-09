"""Progress summary: trends from the local database. Pure code, no LLM call, so it's free and exact."""

from .. import calc, db
from ..state import FitnessState


def progress(state: FitnessState) -> FitnessState:
    profile = state.get("profile") or {}
    if not profile:
        return {"response": "Set up your profile first so I have something to compare against."}

    h = db.history(days=30)
    lines = ["Last 30 days:"]

    current = h["weights"][-1]["value"] if h["weights"] else profile["weight_lb"]
    value = calc.bmi(current, profile["height_in"])
    lines.append(f"- Weight {current:g} lb, BMI {value} ({calc.bmi_category(value)}), goal {profile['goal_weight_lb']:g} lb")
    if len(h["weights"]) >= 2:
        change = current - h["weights"][0]["value"]
        lines.append(f"- Weight change: {change:+.1f} lb since {h['weights'][0]['day']}")

    if h["calories"]:
        recent = h["calories"][-7:]
        avg = sum(d["total"] for d in recent) / len(recent)
        target = calc.calorie_target(profile)["target"]
        lines.append(f"- Average intake over {len(recent)} logged days: {avg:.0f} kcal (target {target})")

    if h["runs"]:
        miles = sum(r["distance_mi"] or 0 for r in h["runs"])
        lines.append(f"- {len(h['runs'])} runs, {miles:.1f} miles total")
        paced = [r for r in h["runs"] if r["distance_mi"] and r["duration_min"]]
        if paced:
            best = min(paced, key=lambda r: r["duration_min"] / r["distance_mi"])
            pace = best["duration_min"] / best["distance_mi"]
            lines.append(f"- Best pace: {int(pace)}:{round(pace % 1 * 60):02d} per mile ({best['day']})")

    if profile.get("max_lifts"):
        maxes = ", ".join(f"{k} {v:g}" for k, v in profile["max_lifts"].items() if v)
        lines.append(f"- Current estimated maxes (lb): {maxes}")

    if len(lines) == 2:
        lines.append("- Not much logged yet. Tell me what you eat, weigh, and train and I'll track it.")
    return {"response": "\n".join(lines)}

"""Plain-text formatting for plans shown in the terminal."""


def format_meal_plan(plan: dict) -> str:
    lines = [f"MEAL PLAN  (target {plan['calorie_target']} kcal)"]
    for meal in plan["meals"]:
        kcal = sum(i["calories"] for i in meal["items"])
        lines.append(f"\n  {meal['name']}  ~{kcal:.0f} kcal")
        for item in meal["items"]:
            lines.append(f"    - {item['amount']} {item['food']}  ({item['calories']:.0f} kcal, {item['protein_g']:.0f} g protein)")
        if meal["swap_options"]:
            lines.append(f"    swaps: {'; '.join(meal['swap_options'])}")
    t, m = plan["totals"], plan["macro_targets"]
    lines.append(
        f"\n  Totals: {t['calories']} kcal | protein {t['protein_g']}/{m['protein_g']} g"
        f" | carbs {t['carbs_g']}/{m['carbs_g']} g | fat {t['fat_g']}/{m['fat_g']} g"
    )
    if plan.get("notes"):
        lines.append(f"  Notes: {plan['notes']}")
    return "\n".join(lines)


def format_workout_plan(plan: dict) -> str:
    weak = ", ".join(plan.get("weak_points") or []) or "none identified"
    lines = [f"WORKOUT PLAN  (weak points: {weak})", f"  {plan['weak_point_strategy']}"]
    for day in plan["days"]:
        if not day["exercises"]:
            lines.append(f"\n  {day['day']}: rest")
            continue
        lines.append(f"\n  {day['day']}: {day['focus']}")
        for ex in day["exercises"]:
            weight = f" @ {ex['weight_lb']:g} lb" if ex.get("weight_lb") else ""
            lines.append(f"    - {ex['name']}: {ex['sets']} x {ex['reps']}{weight}")
    if plan.get("notes"):
        lines.append(f"\n  Notes: {plan['notes']}")
    return "\n".join(lines)

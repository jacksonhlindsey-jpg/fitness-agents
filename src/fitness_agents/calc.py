"""Deterministic fitness math. Agents call these instead of letting the LLM do arithmetic.

Units are imperial at the edges (lb, inches, miles) because that's what users type.
"""

import re

LB_TO_KG = 0.453592
IN_TO_CM = 2.54
KCAL_PER_LB_FAT = 3500

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

# Hard safety limits enforced by the Reviewer, not by the LLM.
CALORIE_FLOOR = {"male": 1500, "female": 1200}
MAX_WEEKLY_LOSS_PCT = 0.01  # at most 1% of body weight per week
MAX_WEEKLY_LOSS_LB = 2.0
MAX_WEEKLY_GAIN_LB = 1.0

# Rough "intermediate" strength standards as a multiple of body weight.
# Only used to compare lifts against each other and find the relative weak point.
STRENGTH_STANDARDS = {
    "male": {"bench": 1.25, "squat": 1.5, "deadlift": 1.75, "ohp": 0.75},
    "female": {"bench": 0.75, "squat": 1.25, "deadlift": 1.5, "ohp": 0.5},
}


def bmi(weight_lb: float, height_in: float) -> float:
    return round(703 * weight_lb / height_in**2, 1)


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "underweight"
    if value < 25:
        return "healthy"
    if value < 30:
        return "overweight"
    return "obese"


def bmr(weight_lb: float, height_in: float, age: int, sex: str) -> float:
    """Mifflin-St Jeor basal metabolic rate in kcal/day."""
    base = 10 * weight_lb * LB_TO_KG + 6.25 * height_in * IN_TO_CM - 5 * age
    return base + (5 if sex == "male" else -161)


def tdee(weight_lb: float, height_in: float, age: int, sex: str, activity: str) -> float:
    return bmr(weight_lb, height_in, age, sex) * ACTIVITY_MULTIPLIERS[activity]


def safe_weekly_rate(weight_lb: float, goal_weight_lb: float, requested_lb_per_week: float) -> float:
    """Clamp the requested weekly change (lb) to a safe range. Negative means losing."""
    requested = abs(requested_lb_per_week)
    if goal_weight_lb < weight_lb:
        cap = min(MAX_WEEKLY_LOSS_LB, weight_lb * MAX_WEEKLY_LOSS_PCT)
        return -min(requested, cap)
    if goal_weight_lb > weight_lb:
        return min(requested, MAX_WEEKLY_GAIN_LB)
    return 0.0


def calorie_target(profile: dict) -> dict:
    """Daily calorie target from the profile, with the rate and the floor both enforced."""
    maintenance = tdee(
        profile["weight_lb"], profile["height_in"], profile["age"], profile["sex"], profile["activity"]
    )
    rate = safe_weekly_rate(profile["weight_lb"], profile["goal_weight_lb"], profile.get("weekly_rate_lb", 1.0))
    target = maintenance + rate * KCAL_PER_LB_FAT / 7
    floor = CALORIE_FLOOR[profile["sex"]]
    return {
        "maintenance": round(maintenance),
        "weekly_rate_lb": round(rate, 2),
        "target": round(max(target, floor)),
        "floor_applied": target < floor,
    }


def macro_targets(calories: float, weight_lb: float) -> dict:
    """Protein 0.8 g/lb, fat 25% of calories, carbs fill the rest."""
    protein = 0.8 * weight_lb
    fat = 0.25 * calories / 9
    carbs = max(calories - protein * 4 - fat * 9, 0) / 4
    return {"protein_g": round(protein), "fat_g": round(fat), "carbs_g": round(carbs)}


def epley_1rm(weight: float, reps: int) -> float:
    """Estimated one-rep max from a set of `reps` at `weight`."""
    if reps <= 1:
        return float(weight)
    return round(weight * (1 + reps / 30), 1)


def strength_profile(max_lifts: dict, weight_lb: float, sex: str) -> dict:
    """Score each lift against its standard and return them weakest-first.

    A score of 1.0 means the lift matches the standard for this body weight.
    Weak points are lifts noticeably behind the user's own average, so the
    result is about balance, not about being "strong enough".
    """
    standards = STRENGTH_STANDARDS[sex]
    scores = {
        lift: round(value / weight_lb / standards[lift], 2)
        for lift, value in max_lifts.items()
        if lift in standards and value
    }
    if len(scores) < 2:  # need at least two lifts to compare
        return {"scores": scores, "weak_points": []}
    average = sum(scores.values()) / len(scores)
    ranked = sorted(scores, key=scores.get)
    weak = [lift for lift in ranked if scores[lift] < average - 0.1] or ranked[:1]
    return {"scores": scores, "weak_points": weak}


def parse_lift(raw: str) -> float:
    """A max lift typed as a one-rep max ("225") or a recent set ("185x5"), as an estimated 1RM."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:[x*]\s*(\d+))?", raw.strip().lower())
    if not m:
        raise ValueError("Use a weight like 225 or a set like 185x5")
    return epley_1rm(float(m.group(1)), int(m.group(2) or 1))

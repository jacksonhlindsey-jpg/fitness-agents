from fitness_agents import calc

PROFILE = {
    "height_in": 70, "weight_lb": 200, "goal_weight_lb": 180, "age": 21,
    "sex": "male", "activity": "moderate", "weekly_rate_lb": 1.0,
}


def test_bmi():
    assert calc.bmi(200, 70) == 28.7
    assert calc.bmi_category(28.7) == "overweight"


def test_bmr_mifflin_st_jeor():
    # 90.72 kg, 177.8 cm, 21 y, male -> 10*90.72 + 6.25*177.8 - 105 + 5
    assert round(calc.bmr(200, 70, 21, "male")) == 1918


def test_calorie_target_deficit():
    t = calc.calorie_target(PROFILE)
    assert t["weekly_rate_lb"] == -1.0
    assert t["target"] == t["maintenance"] - 500


def test_unsafe_rate_is_clamped():
    t = calc.calorie_target({**PROFILE, "weekly_rate_lb": 5})
    assert t["weekly_rate_lb"] == -2.0  # 1% of 200 lb = 2 lb cap


def test_calorie_floor():
    small = {**PROFILE, "sex": "female", "weight_lb": 110, "goal_weight_lb": 100,
             "height_in": 60, "activity": "sedentary", "weekly_rate_lb": 2}
    t = calc.calorie_target(small)
    assert t["target"] == 1200
    assert t["floor_applied"]


def test_epley():
    assert calc.epley_1rm(185, 5) == 215.8
    assert calc.epley_1rm(225, 1) == 225


def test_weak_point_detection():
    s = calc.strength_profile({"bench": 250, "squat": 225, "deadlift": 350, "ohp": 150}, 200, "male")
    assert s["weak_points"] == ["squat"]


def test_single_lift_has_no_weak_point():
    assert calc.strength_profile({"bench": 225}, 200, "male")["weak_points"] == []

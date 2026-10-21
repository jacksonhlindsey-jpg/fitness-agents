"""USDA FoodData Central lookups, so calorie numbers come from real data instead of the LLM.

Docs: https://fdc.nal.usda.gov/api-guide
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from . import config, db

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# FoodData Central nutrient IDs. Energy can appear under several IDs depending on the dataset.
ENERGY_IDS = (1008, 2047, 2048)
NUTRIENT_IDS = {"protein_g": 1003, "fat_g": 1004, "carbs_g": 1005, "fiber_g": 1079}
FILLER_WORDS = {"or", "and", "as", "to", "with", "made", "ns", "of", "in", "a"}


def per_100g(query: str) -> dict | None:
    """Nutrients per 100 g for the best match, or None if nothing usable was found.

    Results (including misses) are cached locally to stay under the API rate limit.
    """
    key = query.strip().lower()
    cached = db.cached_food(key)
    if cached is not db._MISSING:
        return cached

    params = urllib.parse.urlencode({
        "query": key,
        "pageSize": 10,
        "dataType": "Survey (FNDDS),SR Legacy,Foundation",
        "api_key": config.USDA_API_KEY,
    })
    try:
        with urllib.request.urlopen(f"{SEARCH_URL}?{params}", timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None  # network trouble: don't cache, fall back to the LLM estimate this time

    foods = data.get("foods") or []
    nutrients = _extract(_best_match(key, foods)) if foods else None
    db.cache_food(key, nutrients)
    return nutrients


def _words(text: str) -> set[str]:
    words = (w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if w not in FILLER_WORDS}


def _best_match(query: str, foods: list[dict]) -> dict:
    """Pick the result whose description best matches the query.

    The API's top hit is often a processed variant ("Bananas, dehydrated, or
    banana powder" for "banana, raw"), so reward shared words and penalize extras.
    """
    wanted = _words(query)

    def score(food: dict) -> float:
        have = _words(food.get("description", ""))
        return len(wanted & have) - 0.5 * len(have - wanted)

    return max(foods, key=score)


def _extract(food: dict) -> dict | None:
    values = {n.get("nutrientId"): n.get("value") for n in food.get("foodNutrients", [])}
    calories = next((values[i] for i in ENERGY_IDS if values.get(i) is not None), None)
    if calories is None:
        return None
    result = {"description": food.get("description", ""), "calories": calories}
    for name, nutrient_id in NUTRIENT_IDS.items():
        result[name] = values.get(nutrient_id) or 0.0
    return result


def scale(nutrients: dict, grams: float) -> dict:
    factor = grams / 100
    return {
        key: round(nutrients[key] * factor, 1)
        for key in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")
    }

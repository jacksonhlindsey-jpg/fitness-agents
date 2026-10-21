"""USDA FoodData Central lookups, so calorie numbers come from real data instead of the LLM.

Docs: https://fdc.nal.usda.gov/api-guide
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

from . import config, db

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
log = logging.getLogger(__name__)

# FoodData Central nutrient IDs. Energy can appear under several IDs depending on the dataset.
ENERGY_IDS = (1008, 2047, 2048)
NUTRIENT_IDS = {"protein_g": 1003, "fat_g": 1004, "carbs_g": 1005, "fiber_g": 1079}
FILLER_WORDS = {"or", "and", "as", "to", "with", "made", "ns", "of", "in", "a"}
CACHE_PREFIX = "search:"  # cache key namespace for candidate lists


class LookupFailed(RuntimeError):
    """USDA couldn't be reached (network error, bad key, or rate limit)."""


def search(query: str, limit: int = 5) -> list[dict]:
    """Up to `limit` candidate foods with nutrients per 100 g, best guess first.

    The Tracker agent sees all of them and picks the one that matches what the
    user ate. Results (including "no matches") are cached to stay under the
    API's rate limit. Raises LookupFailed if USDA can't be reached.
    """
    key = query.strip().lower()
    cached = db.cached_food(CACHE_PREFIX + key)
    if cached is not db._MISSING:
        return cached or []

    candidates = [c for c in (_extract(f) for f in rank(key, fetch(key))) if c]
    candidates = _dedupe(candidates)[:limit]
    db.cache_food(CACHE_PREFIX + key, candidates)
    return candidates


def fetch(query: str) -> list[dict]:
    """Raw search results from FoodData Central. Raises LookupFailed with the reason.

    Uses POST with a JSON body: USDA's server intermittently rejects GET requests
    whose dataType filter ("Survey (FNDDS)") is in the URL with HTTP 400.
    """
    request = urllib.request.Request(
        f"{SEARCH_URL}?{urllib.parse.urlencode({'api_key': config.USDA_API_KEY})}",
        data=json.dumps({
            "query": query,
            "pageSize": 15,
            "dataType": ["Survey (FNDDS)", "SR Legacy", "Foundation"],
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as resp:
            return json.load(resp).get("foods") or []
    except urllib.error.HTTPError as e:
        log.warning("USDA lookup for %r failed: HTTP %s", query, e.code)  # 403 = bad key, 429 = rate limit
        raise LookupFailed(f"USDA returned HTTP {e.code}") from None
    except urllib.error.URLError as e:
        log.warning("USDA lookup for %r failed: %s", query, e.reason)
        raise LookupFailed(f"USDA could not be reached ({e.reason})") from None
    except (TimeoutError, json.JSONDecodeError) as e:
        log.warning("USDA lookup for %r failed: %s", query, e)
        raise LookupFailed(f"USDA could not be reached ({type(e).__name__})") from None


def _words(text: str) -> set[str]:
    words = (w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if w not in FILLER_WORDS}


def rank(query: str, foods: list[dict]) -> list[dict]:
    """Order results by how well the description matches the query.

    The API's top hit is often a processed variant ("Bananas, dehydrated, or
    banana powder" for "banana, raw"), so reward shared words and penalize extras.
    """
    wanted = _words(query)

    def score(food: dict) -> float:
        have = _words(food.get("description", ""))
        return len(wanted & have) - 0.5 * len(have - wanted)

    return sorted(foods, key=score, reverse=True)


def _extract(food: dict) -> dict | None:
    values = {n.get("nutrientId"): n.get("value") for n in food.get("foodNutrients", [])}
    calories = next((values[i] for i in ENERGY_IDS if values.get(i) is not None), None)
    if calories is None:
        return None
    result = {"description": food.get("description", ""), "calories": round(calories, 1)}
    for name, nutrient_id in NUTRIENT_IDS.items():
        result[name] = round(values.get(nutrient_id) or 0.0, 1)
    return result


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for c in candidates:
        if c["description"].lower() not in seen:
            seen.add(c["description"].lower())
            unique.append(c)
    return unique


def scale(nutrients: dict, grams: float) -> dict:
    factor = grams / 100
    return {
        key: round(nutrients[key] * factor, 1)
        for key in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")
    }

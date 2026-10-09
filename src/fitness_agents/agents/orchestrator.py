"""Orchestrator: reads the user's message and decides which agents should handle it."""

from pydantic import BaseModel

from .. import db, llm
from ..state import FitnessState, Intent

SYSTEM = """You are the Orchestrator of a fitness tracking app. Classify the user's message.

Intents:
- log: the user reports something they ate, drank, weighed, ran, or lifted
- meal_plan: wants a meal plan or calorie target
- workout_plan: wants a workout schedule or program
- full_plan: wants both a meal plan and a workout plan
- progress: asks how they're doing, trends, BMI, or a summary
- chat: anything else (questions, greetings, general fitness questions)

For chat only, also write a short, friendly reply. You are not a doctor: for medical
questions, suggest seeing a professional. For other intents leave reply empty."""

PLANNERS_FOR = {
    "meal_plan": ["nutrition_planner"],
    "workout_plan": ["workout_planner"],
    "full_plan": ["nutrition_planner", "workout_planner"],
}


class Routing(BaseModel):
    intent: Intent
    reply: str


def orchestrator(state: FitnessState) -> FitnessState:
    routing = llm.ask(SYSTEM, state["user_input"], Routing, effort="low")
    update: FitnessState = {
        "intent": routing.intent,
        "profile": db.get_profile() or {},
        "planners": PLANNERS_FOR.get(routing.intent, []),
        "revisions": 0,
    }
    if routing.intent == "chat":
        update["response"] = routing.reply
    return update

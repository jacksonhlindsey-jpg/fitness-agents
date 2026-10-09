"""The LangGraph workflow that connects the agents.

    user message
        │
    Orchestrator ──► Tracker ─────────────► end
        │      └───► Progress ────────────► end
        │      └───► (chat reply) ────────► end
        ▼
    Nutrition Planner ┐   (run in parallel when both are needed)
    Workout Planner  ─┴► Reviewer ──fails──► back to the failing planner(s), up to MAX_REVISIONS
                            │
                          passes
                            ▼
                     Human approval  ──edit──► back to the planner(s) with the user's feedback
                            │
                         approve
                            ▼
                        Save plans ──► end
"""

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import config, db
from .agents import nutrition_planner, orchestrator, progress, reviewer, tracker, workout_planner
from .state import FitnessState

PLANNER_FOR = {"meal": "nutrition_planner", "workout": "workout_planner"}


def route_intent(state: FitnessState):
    intent = state["intent"]
    if intent == "log":
        return "tracker"
    if intent == "progress":
        return "progress"
    if state["planners"]:
        return state["planners"] if state.get("profile") else "needs_profile"
    return END  # chat: the Orchestrator already wrote the reply


def needs_profile(state: FitnessState) -> FitnessState:
    return {"response": "I need your profile before I can build a plan. Type /profile to set it up."}


def route_review(state: FitnessState):
    """Send failing plans back for another draft, or move on to the user."""
    review = state["review"]
    failing = [PLANNER_FOR[kind] for kind, issues in review.items() if issues and state.get(f"{kind}_plan")]
    if failing and state["revisions"] <= config.MAX_REVISIONS:
        return failing
    return "human_approval"


def human_approval(state: FitnessState) -> Command[Literal["save_plans", "nutrition_planner", "workout_planner", "__end__"]]:
    """Pause the graph and let the user approve, edit, or reject the plan.

    The CLI resumes with {"action": "approve"}, {"action": "reject"}, or
    {"action": "edit", "targets": ["meal" | "workout", ...], "feedback": "..."}.
    """
    decision = interrupt({
        "meal_plan": state.get("meal_plan"),
        "workout_plan": state.get("workout_plan"),
        "unresolved_issues": state.get("review", {}),
    })
    if decision["action"] == "approve":
        return Command(goto="save_plans")
    if decision["action"] == "edit":
        return Command(
            update={"user_feedback": decision["feedback"], "review": {}, "revisions": 0},
            goto=[PLANNER_FOR[t] for t in decision["targets"]],
        )
    return Command(update={"response": "Okay, I discarded that plan."}, goto=END)


def save_plans(state: FitnessState) -> FitnessState:
    saved = []
    for kind in ("meal", "workout"):
        if plan := state.get(f"{kind}_plan"):
            db.save_plan(kind, plan)
            saved.append(kind)
    return {"approved": True, "response": f"Saved your {' and '.join(saved)} plan."}


def build_graph():
    g = StateGraph(FitnessState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("tracker", tracker)
    g.add_node("progress", progress)
    g.add_node("needs_profile", needs_profile)
    g.add_node("nutrition_planner", nutrition_planner)
    g.add_node("workout_planner", workout_planner)
    g.add_node("reviewer", reviewer)
    g.add_node("human_approval", human_approval)
    g.add_node("save_plans", save_plans)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator", route_intent,
        ["tracker", "progress", "needs_profile", "nutrition_planner", "workout_planner", END],
    )
    g.add_edge("nutrition_planner", "reviewer")
    g.add_edge("workout_planner", "reviewer")
    g.add_conditional_edges("reviewer", route_review, ["nutrition_planner", "workout_planner", "human_approval"])
    for node in ("tracker", "progress", "needs_profile", "save_plans"):
        g.add_edge(node, END)

    # The checkpointer is what lets the graph pause at human_approval and resume later.
    return g.compile(checkpointer=InMemorySaver())

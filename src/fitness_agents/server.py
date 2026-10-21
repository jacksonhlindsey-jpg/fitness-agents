"""Web API for the fitness agents, plus the web UI in ../web for local use.

Run with:  uv run fitness-web   (then open http://localhost:8000)

The agents and the LangGraph workflow are the same ones the CLI uses. A plan
request pauses at the human approval step and returns a thread_id; the UI
resumes it with POST /api/approve.
"""

import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from . import calc, config, db, llm
from .graph import build_graph

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Fitness Agents")
# The hosted UI (GitHub Pages) lives on a different origin than this API.
origins = os.getenv("FITNESS_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# One compiled graph for the process: its checkpointer holds paused plan requests by thread_id.
graph = build_graph()


# --- profile -------------------------------------------------------------------

class ProfileIn(BaseModel):
    height_in: float = Field(gt=36, lt=96)
    weight_lb: float = Field(gt=60, lt=700)
    goal_weight_lb: float = Field(gt=60, lt=700)
    age: int = Field(ge=13, le=100)
    sex: Literal["male", "female"]
    activity: Literal["sedentary", "light", "moderate", "active", "very_active"]
    weekly_rate_lb: float = Field(default=1.0, ge=0, le=5)
    food_likes: str = ""
    food_avoid: str = ""
    training_days: int = Field(default=3, ge=1, le=7)
    equipment: str = "full gym"
    avoid_exercises: str = ""
    max_lifts: dict[Literal["bench", "squat", "deadlift", "ohp"], str] = {}


def summary(profile: dict) -> dict:
    """The numbers every view needs, computed from the profile."""
    target = calc.calorie_target(profile)
    bmi = calc.bmi(profile["weight_lb"], profile["height_in"])
    return {
        "bmi": bmi,
        "bmi_category": calc.bmi_category(bmi),
        "calories": target,
        "macros": calc.macro_targets(target["target"], profile["weight_lb"]),
        "water_goal_oz": round(profile["weight_lb"] * 0.5),  # common rule of thumb
    }


@app.get("/api/profile")
def get_profile():
    profile = db.get_profile()
    return {"profile": profile, "summary": summary(profile) if profile else None}


@app.put("/api/profile")
def put_profile(body: ProfileIn):
    profile = body.model_dump()
    try:
        profile["max_lifts"] = {
            lift: calc.parse_lift(raw) for lift, raw in body.max_lifts.items() if raw.strip()
        }
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    db.save_profile(profile)
    return {"profile": profile, "summary": summary(profile)}


# --- dashboards ----------------------------------------------------------------

@app.get("/api/today")
def get_today():
    profile = db.get_profile()
    return {
        "date": db.today(),
        "totals": db.day_totals(),
        "entries": db.day_entries(),
        "summary": summary(profile) if profile else None,
    }


@app.get("/api/progress")
def get_progress(days: int = 30):
    profile = db.get_profile()
    if not profile:
        raise HTTPException(404, "Set up your profile first.")
    h = db.history(days)
    weights = [
        {"day": w["day"], "weight_lb": w["value"], "bmi": calc.bmi(w["value"], profile["height_in"])}
        for w in h["weights"]
    ]
    current = weights[-1]["weight_lb"] if weights else profile["weight_lb"]
    return {
        "weights": weights,
        "current_weight_lb": current,
        "current_bmi": calc.bmi(current, profile["height_in"]),
        "goal_weight_lb": profile["goal_weight_lb"],
        "calories": h["calories"],
        "calorie_target": calc.calorie_target(profile)["target"],
        "runs": h["runs"],
        "max_lifts": profile.get("max_lifts", {}),
    }


@app.get("/api/plans")
def get_plans():
    return {"meal_plan": db.latest_plan("meal"), "workout_plan": db.latest_plan("workout")}


# --- chat with the agents ------------------------------------------------------

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ApproveIn(BaseModel):
    thread_id: str
    action: Literal["approve", "edit", "reject"]
    targets: list[Literal["meal", "workout"]] = []
    feedback: str = ""


def _run(payload, thread_id: str) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(payload, cfg)
    except llm.LLMError as e:
        raise HTTPException(503, str(e)) from None
    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        return {"type": "approval", "thread_id": thread_id, **pending}
    return {"type": "message", "response": result.get("response", "")}


@app.post("/api/chat")
def chat(body: ChatIn):
    return _run({"user_input": body.message}, str(uuid.uuid4()))


@app.post("/api/approve")
def approve(body: ApproveIn):
    if not graph.get_state({"configurable": {"thread_id": body.thread_id}}).next:
        raise HTTPException(404, "That plan is no longer waiting for approval. Ask for a new one.")
    if body.action == "edit" and not (body.targets and body.feedback.strip()):
        raise HTTPException(422, "Say what to change and which plan to change.")
    return _run(Command(resume=body.model_dump(exclude={"thread_id"})), body.thread_id)


@app.get("/api/status")
def status():
    return {"provider": config.PROVIDER, "model": config.MODEL}


# Serve the UI last so /api routes take priority.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"Fitness Agents web UI: http://localhost:{port}")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=port)

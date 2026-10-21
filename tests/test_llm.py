"""Gemini retry and fallback, with a fake client (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from fitness_agents import config, llm


class Answer(BaseModel):
    ok: bool


def overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def quota():
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})


class FakeModels:
    def __init__(self, failures):
        self.failures = failures  # model name -> list of exceptions to raise, in order
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        queue = self.failures.get(model, [])
        if queue:
            raise queue.pop(0)
        return SimpleNamespace(text='{"ok": true}')


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "gemini")
    monkeypatch.setattr(config, "MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(config, "FAST_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def install(failures):
        models = FakeModels(failures)
        monkeypatch.setattr(llm, "_gemini_client", SimpleNamespace(models=models))
        return models

    return install


def test_retries_same_model_when_briefly_overloaded(fake_gemini):
    models = fake_gemini({"gemini-3.8-flash": [overloaded()]})
    assert llm.ask("sys", "hi", Answer, effort="medium").ok
    assert models.calls == ["gemini-3.8-flash", "gemini-3.8-flash"]


def test_falls_back_to_next_model(fake_gemini):
    models = fake_gemini({"gemini-3.8-flash": [overloaded()] * 3, "gemini-3.6-flash": [quota()]})
    assert llm.ask("sys", "hi", Answer, effort="medium").ok
    assert models.calls == ["gemini-3.8-flash"] * 3 + ["gemini-3.6-flash", "gemini-3.5-flash-lite"]


def test_gives_up_with_a_helpful_message(fake_gemini):
    fake_gemini({m: [overloaded()] * 3 for m in ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")})
    with pytest.raises(llm.LLMError, match="ollama"):
        llm.ask("sys", "hi", Answer, effort="medium")


def test_quick_jobs_use_the_fast_model_first(fake_gemini):
    models = fake_gemini({"gemini-3.5-flash-lite": [quota()]})
    assert llm.ask("sys", "hi", Answer, effort="low").ok
    assert models.calls == ["gemini-3.5-flash-lite", "gemini-3.6-flash"]

"""Thin wrapper around the Anthropic SDK used by every agent.

All agent calls go through `ask()`, which returns a validated Pydantic object
(structured outputs), so agents never have to parse free-form JSON.
"""

from typing import Literal, TypeVar

import anthropic
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high"]

# Models that accept the server-side refusal fallback ("default" routing).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client: anthropic.Anthropic | None = None


class LLMError(RuntimeError):
    """The model did not return a usable answer."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def ask(system: str, prompt: str, schema: type[T], effort: Effort = "low") -> T:
    """Send one request and return the response parsed into `schema`."""
    extra = {}
    if config.MODEL in _FALLBACK_MODELS:
        extra = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}

    response = _get_client().beta.messages.parse(
        model=config.MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        output_config={"effort": effort},
        **extra,
    )

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("The response was cut off before it finished.")
    if response.parsed_output is None:
        raise LLMError("The response did not match the expected format.")
    return response.parsed_output

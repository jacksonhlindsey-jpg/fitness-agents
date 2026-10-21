"""The one place agents talk to an LLM. Every call returns a validated Pydantic object.

Providers (set FITNESS_PROVIDER, or let it pick automatically):
- gemini:    Google Gemini free tier. Needs a free key from https://aistudio.google.com/apikey
- ollama:    a model running locally through Ollama. Free, no key, data never leaves the computer.
- anthropic: Claude. Paid API, optional.

Default: gemini when GEMINI_API_KEY is set, otherwise ollama.
"""

import json
import urllib.error
import urllib.request
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    """The model did not return a usable answer. The message is safe to show the user."""


def ask(system: str, prompt: str, schema: type[T], effort: Effort = "low") -> T:
    """Send one request and return the response parsed into `schema`."""
    providers = {"gemini": _ask_gemini, "ollama": _ask_ollama, "anthropic": _ask_anthropic}
    text = providers[config.PROVIDER](system, prompt, schema, effort)
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        raise LLMError("The model's answer didn't match the expected format. Try again.") from None


# --- Gemini (free tier) ------------------------------------------------------

_gemini_client = None


def _ask_gemini(system: str, prompt: str, schema: type[BaseModel], effort: Effort) -> str:
    from google import genai
    from google.genai import errors, types

    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    try:
        response = _gemini_client.models.generate_content(
            model=config.MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
    except errors.ClientError as e:
        if e.code == 429:
            raise LLMError("Hit the Gemini free-tier rate limit. Wait a minute and try again.") from None
        if e.code in (400, 401, 403) and "key" in str(e).lower():
            raise LLMError("Your GEMINI_API_KEY is missing or invalid. Check .env.") from None
        raise LLMError(f"Gemini rejected the request ({e.code}).") from None
    except errors.ServerError:
        raise LLMError("Gemini is having trouble right now. Try again in a moment.") from None
    if not response.text:
        raise LLMError("Gemini returned an empty answer (it may have been blocked). Try rephrasing.")
    return response.text


# --- Ollama (local, free) ----------------------------------------------------

def _ask_ollama(system: str, prompt: str, schema: type[BaseModel], effort: Effort) -> str:
    body = {
        "model": config.MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "format": schema.model_json_schema(),  # constrains the output to this JSON schema
        "stream": False,
        "think": False,  # skip slow step-by-step reasoning on models that support it
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if "not found" in detail:
            raise LLMError(f"Ollama doesn't have {config.MODEL}. Run: ollama pull {config.MODEL}") from None
        raise LLMError(f"Ollama returned an error: {detail[:200]}") from None
    except (urllib.error.URLError, ConnectionError):
        raise LLMError("Can't reach Ollama. Start it with `ollama serve` (or open the Ollama app).") from None
    except TimeoutError:
        raise LLMError("The local model took too long. Try a smaller model.") from None
    return data["message"]["content"]


# --- Anthropic (optional, paid) ----------------------------------------------

_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
_anthropic_client = None


def _ask_anthropic(system: str, prompt: str, schema: type[BaseModel], effort: Effort) -> str:
    import anthropic

    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    extra = {}
    if config.MODEL in _FALLBACK_MODELS:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        response = _anthropic_client.beta.messages.parse(
            model=config.MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            **extra,
        )
    except anthropic.AuthenticationError:
        raise LLMError("Your ANTHROPIC_API_KEY is missing or invalid. Check .env.") from None
    except anthropic.RateLimitError:
        raise LLMError("Hit the Anthropic rate limit. Wait a moment and try again.") from None
    except anthropic.APIError as e:
        raise LLMError(f"The Claude API returned an error ({e.__class__.__name__}).") from None
    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise LLMError("The response was cut off or malformed. Try again.")
    return response.parsed_output.model_dump_json()

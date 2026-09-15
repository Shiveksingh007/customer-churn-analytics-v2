"""Shared Gemini client for the GenAI layer."""

from __future__ import annotations

import os
import time

from utils.env_loader import get_gemini_api_key, load_project_env

# Prefer Flash Lite / 2.5 Flash — gemini-2.0-flash free-tier often shows limit: 0.
_DEFAULT_MODEL = "gemini-2.5-flash"
_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-1.5-flash",
)


def get_gemini_model() -> str:
    load_project_env()
    return (os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL).strip()


GEMINI_MODEL = _DEFAULT_MODEL


def call_llm(system: str, user: str, *, max_output_tokens: int = 2048) -> str:
    """
    Call Gemini with a system instruction + user prompt.
    Returns plain text (callers parse JSON themselves).

    On free-tier quota exhaustion for one model, retries once with a fallback model.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=get_gemini_api_key())
    models_to_try = [get_gemini_model()]
    for candidate in _FALLBACK_MODELS:
        if candidate not in models_to_try:
            models_to_try.append(candidate)

    last_error: Exception | None = None
    for index, model in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                ),
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text.strip()
        except Exception as exc:  # noqa: BLE001 — map provider errors clearly
            last_error = exc
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                # Brief pause then try next model (free-tier model lockouts are common).
                if index < len(models_to_try) - 1:
                    time.sleep(1.5)
                    continue
                raise RuntimeError(
                    "Gemini quota exhausted (or free-tier limit is 0 for this model). "
                    "This is usually NOT because you made many requests — Google often "
                    "assigns limit:0 to gemini-2.0-flash on free tier. "
                    "Fix: (1) set GEMINI_MODEL=gemini-2.5-flash in app/.env, "
                    "(2) create a key at https://aistudio.google.com/apikey, "
                    "(3) or enable billing / wait for daily reset. "
                    f"Original error: {exc}"
                ) from exc
            raise

    raise RuntimeError(f"Gemini call failed: {last_error}")

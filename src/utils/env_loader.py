"""Load environment variables from project .env files."""

from __future__ import annotations

from pathlib import Path

_PLACEHOLDER_KEYS = frozenset(
    {
        "sk-ant-your-key-here",
        "your-api-key-here",
        "paste-your-key-here",
        "your-gemini-api-key-here",
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    """
    Load ``.env`` from repo root and ``app/.env`` if present.

    Existing shell environment variables take precedence (override=False).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root = project_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / "app" / ".env", override=False)


def get_gemini_api_key() -> str:
    """Return a validated Gemini API key or raise a helpful error."""
    import os

    load_project_env()
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        try:
            import streamlit as st
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to app/.env or export it in your shell. "
            "Get a key from https://aistudio.google.com/apikey"
        )
    if api_key in _PLACEHOLDER_KEYS:
        raise EnvironmentError(
            "GEMINI_API_KEY is still the placeholder value. "
            "Replace it with a real key from https://aistudio.google.com/apikey"
        )
    return api_key

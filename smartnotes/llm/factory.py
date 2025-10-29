from __future__ import annotations
from typing import Optional

from smartnotes.config import load_settings
from .local import LocalLLM
from .openai import OpenAILLM  # type: ignore
from .base import LLMProvider


def get_provider(backend: Optional[str] = None) -> LLMProvider:
    s = load_settings()
    name = (backend or s.llm_backend or "local").lower()

    # Honor remote_allowed: if false, force local
    if not s.remote_allowed:
        name = "local"

    if name == "local":
        return LocalLLM()
    if name == "openai":
        return OpenAILLM()

    # Placeholder for future: openai, anthropic, gemini
    # For now, fall back to local to be safe
    return LocalLLM()

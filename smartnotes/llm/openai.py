from __future__ import annotations
import os
import json
from typing import Any, List

from .base import LLMProvider


class OpenAILLM(LLMProvider):
    """OpenAI provider using chat.completions API.

    Requires OPENAI_API_KEY to be set in the environment.
    Model can be overridden via SMARTNOTES_OPENAI_MODEL; defaults to 'gpt-4o-mini'.
    """

    def __init__(self, model: str | None = None, timeout: float = 20.0):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai package not installed. Install with: uv sync --group llm") from e

        self._client = OpenAI()
        self._model = model or os.getenv("SMARTNOTES_OPENAI_MODEL") or "gpt-4o-mini"
        self._timeout = timeout

    def _chat(self, system_msg: str, user_msg: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=self._timeout,  # type: ignore[arg-type]
            )
            content = (resp.choices[0].message.content or "").strip()
            return content
        except Exception:
            return ""

    def generate_summary(self, text: str, *, max_sentences: int = 5, **kwargs: Any) -> str:
        if not text or not text.strip():
            return ""
        system = (
            "You are a succinct assistant that produces short, faithful summaries. "
            "Write at most the requested number of sentences; no extra commentary."
        )
        user = f"Summarize the following note in at most {max_sentences} sentences:\n\n{text}"
        return self._chat(system, user)

    def generate_tags(self, text: str, *, top_k: int = 5, **kwargs: Any) -> List[str]:
        if not text or not text.strip():
            return []
        system = (
            "You extract topical tags for notes. Rules: lowercase words/phrases, no punctuation, 1-3 words each. "
            "Return ONLY a JSON array of strings (e.g., [\"productivity\", \"mindset\"])."
        )
        user = f"Extract up to {top_k} tags from the note below. Respond with JSON only.\n\n{text}"
        out = self._chat(system, user)
        # Try parsing JSON response
        try:
            data = json.loads(out)
            if isinstance(data, list):
                cleaned: List[str] = []
                for t in data:
                    if not isinstance(t, str):
                        continue
                    t = t.strip().lower()
                    if t and t not in cleaned:
                        cleaned.append(t)
                return cleaned[:top_k]
        except Exception:
            pass
        # Fallback: naive keyword extraction
        import re
        words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        tags = sorted(freq.keys(), key=lambda w: (-freq[w], w))[:top_k]
        return tags

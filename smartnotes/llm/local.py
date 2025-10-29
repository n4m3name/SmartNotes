from __future__ import annotations
from typing import Any, List
import re

from .base import LLMProvider


class LocalLLM(LLMProvider):
    """Naive local provider: rule-based summary and tags (no network)."""

    def generate_summary(self, text: str, *, max_sentences: int = 5, **kwargs: Any) -> str:
        # Extract first N sentences as a crude summary
        # Split on . ! ? followed by whitespace/newline
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        sents = [s.strip() for s in sents if s.strip()]
        return " ".join(sents[:max_sentences])

    def generate_tags(self, text: str, *, top_k: int = 5, **kwargs: Any) -> List[str]:
        # Extract simple lowercase keywords (alnum-only) by frequency, ignoring very short tokens
        words = re.findall(r"[A-Za-z0-9]{4,}", text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        tags = sorted(freq.keys(), key=lambda w: (-freq[w], w))[:top_k]
        return tags

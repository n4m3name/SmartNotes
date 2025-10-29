from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def generate_summary(self, text: str, *, max_sentences: int = 5, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_tags(self, text: str, *, top_k: int = 5, **kwargs: Any) -> List[str]:
        raise NotImplementedError

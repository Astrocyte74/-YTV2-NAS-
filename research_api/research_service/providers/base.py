"""Provider interfaces and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import urlparse

from ..models import ResearchBatchResult


def extract_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def canonical_domain(domain: str) -> str:
    clean = (domain or "").strip().lower()
    for prefix in ("www.", "m.", "amp."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    return clean


class ResearchProvider(ABC):
    name: str

    @abstractmethod
    def execute(
        self,
        *,
        query: str,
        tool: str,
        max_results: int,
        options: dict | None = None,
    ) -> ResearchBatchResult:
        raise NotImplementedError

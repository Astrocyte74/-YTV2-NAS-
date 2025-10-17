"""Web page fetcher built on top of the layered extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..web_extract import WebPageContent, WebPageExtractor


@dataclass
class WebFetcherResult:
    """Structured fetch result suitable for summarizer ingestion."""

    content: WebPageContent
    text: str


class WebFetcherError(Exception):
    """Raised when a web page cannot be fetched or normalised."""


class WebPageFetcher:
    """Thin wrapper around :class:`WebPageExtractor` with error translation."""

    def __init__(self, allow_dynamic: bool = False) -> None:
        self._extractor = WebPageExtractor(allow_dynamic=allow_dynamic)

    def fetch(self, url: str) -> WebFetcherResult:
        if not url:
            raise WebFetcherError("Empty URL provided.")
        try:
            content = self._extractor.extract(url)
        except Exception as exc:  # pragma: no cover - network errors
            raise WebFetcherError(str(exc)) from exc
        if not content.text or len(content.text.strip()) < 200:
            raise WebFetcherError("Page does not contain enough extractable text.")
        return WebFetcherResult(content=content, text=content.text)

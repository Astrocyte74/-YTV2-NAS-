"""Web page fetcher built on top of the layered extractor."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from ..web_extract import WebPageContent, WebPageExtractor

logger = logging.getLogger(__name__)


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

    def _try_tavily_extract(self, url: str) -> Optional[WebFetcherResult]:
        """Try Tavily extract API as a fallback when primary extraction fails.

        Gated by TAVILY_EXTRACT_FALLBACK env var (default enabled).
        Returns None if Tavily is unavailable or extraction fails.
        """
        if os.getenv("TAVILY_EXTRACT_FALLBACK", "1").strip().lower() in ("0", "off", "false"):
            return None

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            import requests as _req

            logger.info("Attempting Tavily extract fallback for %s", url)
            resp = _req.post(
                "https://api.tavily.com/extract",
                json={
                    "api_key": api_key,
                    "urls": [url],
                    "extract_depth": "advanced",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            results = data.get("results") or []
            if not results:
                logger.info("Tavily extract returned no results for %s", url)
                return None

            item = results[0]
            text = str(item.get("raw_content") or item.get("content") or "").strip()
            if len(text) < 200:
                logger.info("Tavily extract returned short text (%d chars) for %s", len(text), url)
                return None

            logger.info("Tavily extract succeeded for %s (%d chars)", url, len(text))
            return WebFetcherResult(
                content=WebPageContent(
                    source_url=url,
                    canonical_url=str(item.get("url") or "").strip() or url,
                    title=str(item.get("title") or "").strip() or url,
                    text=text,
                    language=None,
                    extractor_notes={"final_method": "tavily_extract"},
                ),
                text=text,
            )
        except Exception as exc:
            logger.warning("Tavily extract fallback failed for %s: %s", url, exc)
            return None

    def fetch(self, url: str) -> WebFetcherResult:
        if not url:
            raise WebFetcherError("Empty URL provided.")
        try:
            content = self._extractor.extract(url)
        except Exception as exc:  # pragma: no cover - network errors
            # Try Tavily fallback before giving up
            tavily_result = self._try_tavily_extract(url)
            if tavily_result:
                return tavily_result
            raise WebFetcherError(str(exc)) from exc
        if not content.text or len(content.text.strip()) < 200:
            # Try Tavily fallback for short-text cases
            tavily_result = self._try_tavily_extract(url)
            if tavily_result:
                return tavily_result
            notes = getattr(content, "extractor_notes", None) or {}
            if isinstance(notes, dict):
                urlctx = str(notes.get("url_context") or "").strip()
                if urlctx:
                    hint = ""
                    if "readtimeout" in urlctx.lower():
                        hint = " (try increasing WEB_URL_CONTEXT_TIMEOUT / WEB_URL_CONTEXT_PDF_TIMEOUT)"
                    raise WebFetcherError(f"Page does not contain enough extractable text; URL-context: {urlctx}{hint}.")
            raise WebFetcherError("Page does not contain enough extractable text.")
        return WebFetcherResult(content=content, text=content.text)

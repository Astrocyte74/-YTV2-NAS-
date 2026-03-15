"""Brave Search provider adapter."""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests

from ..config import (
    BRAVE_API_KEY,
    BRAVE_DEFAULT_COUNTRY,
    BRAVE_DEFAULT_SEARCH_LANG,
    BRAVE_DEFAULT_UI_LANG,
    BRAVE_MIN_INTERVAL_SECONDS,
    BRAVE_NEWS_URL,
    BRAVE_RETRY_429_DELAY_SECONDS,
    BRAVE_RETRY_429_MAX_ATTEMPTS,
    BRAVE_SAFESEARCH,
    BRAVE_WEB_URL,
)
from ..models import ResearchBatchResult, ResearchItem
from .base import ResearchProvider

_LAST_BRAVE_CALL_TS = 0.0
_BRAVE_NEXT_ALLOWED_TS = 0.0
_BRAVE_THROTTLE_LOCK = threading.Lock()
_FRESHNESS_PAST_DAY_HINTS = (
    "today",
    "latest",
    "current",
    "right now",
    "breaking",
    "just announced",
    "this morning",
    "this afternoon",
    "this evening",
)
_FRESHNESS_PAST_WEEK_HINTS = (
    "recent",
    "new",
    "updates",
    "update",
    "this week",
    "launch date",
    "release date",
    "release notes",
)


def _throttle_brave_call() -> None:
    global _LAST_BRAVE_CALL_TS
    with _BRAVE_THROTTLE_LOCK:
        now = time.monotonic()
        next_allowed = max(_LAST_BRAVE_CALL_TS + BRAVE_MIN_INTERVAL_SECONDS, _BRAVE_NEXT_ALLOWED_TS)
        wait_for = next_allowed - now
        if wait_for > 0:
            time.sleep(wait_for)
        _LAST_BRAVE_CALL_TS = time.monotonic()


def _header_number(resp: requests.Response, name: str) -> float | None:
    value = (resp.headers or {}).get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_rate_limit_state(resp: requests.Response) -> None:
    global _BRAVE_NEXT_ALLOWED_TS
    remaining = _header_number(resp, "X-RateLimit-Remaining")
    reset = _header_number(resp, "X-RateLimit-Reset")
    if remaining is None or reset is None or remaining > 0 or reset <= 0:
        return
    with _BRAVE_THROTTLE_LOCK:
        _BRAVE_NEXT_ALLOWED_TS = max(_BRAVE_NEXT_ALLOWED_TS, time.monotonic() + reset)


def _retry_delay_for_429(resp: requests.Response, attempts: int) -> float:
    retry_after = (resp.headers or {}).get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), BRAVE_RETRY_429_DELAY_SECONDS)
        except (TypeError, ValueError):
            pass
    reset = _header_number(resp, "X-RateLimit-Reset")
    if reset is not None and reset > 0:
        return max(reset, BRAVE_RETRY_429_DELAY_SECONDS)
    multiplier = max(0, attempts - 1)
    return (BRAVE_RETRY_429_DELAY_SECONDS * (2**multiplier)) + random.uniform(0.05, 0.35)


def _query_looks_english(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _choose_freshness(query: str, tool: str, options: dict[str, Any]) -> str | None:
    if not query:
        return None
    requested = str(options.get("freshness") or "").strip().lower()
    if requested in {"on", "pd", "pw", "pm", "py"}:
        return requested
    if tool == "news":
        return "pw"
    lower_query = f" {(query or '').lower()} "
    if any(hint in lower_query for hint in _FRESHNESS_PAST_DAY_HINTS):
        return "pd"
    if any(hint in lower_query for hint in _FRESHNESS_PAST_WEEK_HINTS):
        return "pw"
    if not options.get("freshness_sensitive"):
        return None
    return "pm"


def _build_params(
    *,
    query: str,
    tool: str,
    max_results: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": query,
        "count": max(1, min(max_results, 20)),
    }

    freshness = _choose_freshness(query, tool, options)
    if freshness:
        params["freshness"] = freshness

    country = str(options.get("country") or BRAVE_DEFAULT_COUNTRY).strip().upper()
    if country:
        params["country"] = country

    if _query_looks_english(query):
        search_lang = str(options.get("search_lang") or BRAVE_DEFAULT_SEARCH_LANG).strip().lower()
        if search_lang:
            params["search_lang"] = search_lang
        if tool == "news":
            ui_lang = str(options.get("ui_lang") or BRAVE_DEFAULT_UI_LANG).strip()
            if ui_lang:
                params["ui_lang"] = ui_lang

    safesearch = str(options.get("safesearch") or BRAVE_SAFESEARCH).strip().lower()
    if safesearch in {"off", "moderate", "strict"}:
        params["safesearch"] = safesearch

    if tool in {"web", "news"} and (
        bool(options.get("extra_snippets")) or str(options.get("depth") or "").strip().lower() in {"balanced", "deep"}
    ):
        params["extra_snippets"] = "true"

    return params


def _combine_snippet(row: dict[str, Any]) -> str:
    base = str(row.get("description") or row.get("snippet") or "").strip()
    extras = row.get("extra_snippets") or []
    if not isinstance(extras, list):
        return base
    extra_parts: list[str] = []
    for value in extras[:2]:
        text = str(value or "").strip()
        if text and text not in extra_parts and text != base:
            extra_parts.append(text)
    if not extra_parts:
        return base
    if not base:
        return " ".join(extra_parts)
    return f"{base} {' '.join(extra_parts)}".strip()


class BraveProvider(ResearchProvider):
    name = "brave"

    def execute(
        self,
        *,
        query: str,
        tool: str,
        max_results: int,
        options: dict | None = None,
    ) -> ResearchBatchResult:
        started = time.time()
        result = ResearchBatchResult(query=query, provider=self.name, tool=tool)
        options = dict(options or {})

        if not BRAVE_API_KEY:
            result.errors.append("BRAVE_API_KEY not configured")
            return result

        endpoint = BRAVE_WEB_URL
        if tool == "news":
            endpoint = BRAVE_NEWS_URL

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        }

        params = _build_params(query=query, tool=tool, max_results=max_results, options=options)

        try:
            resp = None
            attempts = 0
            max_attempts = max(0, BRAVE_RETRY_429_MAX_ATTEMPTS)
            while True:
                _throttle_brave_call()
                resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
                _update_rate_limit_state(resp)
                if resp.status_code != 429:
                    break

                result.rate_limited = True
                if attempts >= max_attempts:
                    break
                attempts += 1
                result.retry_count = attempts
                delay = _retry_delay_for_429(resp, attempts)
                time.sleep(delay)

            if resp is None:
                raise RuntimeError("brave request not attempted")

            resp.raise_for_status()
            data = resp.json() or {}

            if endpoint == BRAVE_NEWS_URL:
                news = ((data.get("results") or []))
                for row in news[:max_results]:
                    url = str(row.get("url") or "").strip()
                    if not url:
                        continue
                    result.results.append(
                        ResearchItem(
                            title=str(row.get("title") or "Untitled"),
                            url=url,
                            snippet=_combine_snippet(row),
                            published_at=str(row.get("age") or row.get("published") or "") or None,
                        )
                    )
            else:
                web = ((data.get("web") or {}).get("results") or [])
                for row in web[:max_results]:
                    url = str(row.get("url") or "").strip()
                    if not url:
                        continue
                    result.results.append(
                        ResearchItem(
                            title=str(row.get("title") or "Untitled"),
                            url=url,
                            snippet=_combine_snippet(row),
                            published_at=str(row.get("age") or "") or None,
                        )
                    )
        except Exception as exc:
            result.errors.append(f"brave {tool} failed: {exc}")

        result.latency_ms = int((time.time() - started) * 1000)
        return result

"""Tavily provider adapter."""

from __future__ import annotations

import re
import time
from urllib.parse import urlparse

import requests

from ..config import (
    TAVILY_API_KEY,
    TAVILY_CRAWL_URL,
    TAVILY_EXTRACT_URL,
    TAVILY_MAP_URL,
    TAVILY_RESEARCH_MODEL,
    TAVILY_RESEARCH_POLL_SECONDS,
    TAVILY_RESEARCH_TIMEOUT_SECONDS,
    TAVILY_RESEARCH_URL,
    TAVILY_SEARCH_URL,
)
from ..models import ResearchBatchResult, ResearchItem
from .base import ResearchProvider, extract_domain

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s)\]>\"']*)?",
    re.IGNORECASE,
)
_TAVILY_RESEARCH_AVAILABLE = True
_TAVILY_RESEARCH_DISABLED_REASON = ""
_GENERIC_PAGE_HINTS = (
    "support",
    "help",
    "home",
    "landing",
    "application support",
    "software updates",
    "contact",
    "repair",
    "products",
)


def _truncate(text: str, limit: int = 900) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3].rstrip()}..."


def _dedupe_items(items: list[ResearchItem], limit: int) -> list[ResearchItem]:
    out: list[ResearchItem] = []
    seen: set[str] = set()
    for item in items:
        url = (item.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _looks_like_url(value: str) -> bool:
    return bool(_URL_RE.fullmatch((value or "").strip()))


def _normalize_site_target(value: str) -> str:
    clean = (value or "").strip().rstrip(".,;:)]}>")
    if not clean:
        return ""
    if _looks_like_url(clean):
        return clean
    if _DOMAIN_RE.fullmatch(clean):
        return f"https://{clean}"
    return ""


def _title_from_url(url: str) -> str:
    parsed = urlparse(url or "")
    path = (parsed.path or "/").strip("/")
    if not path:
        return extract_domain(url) or "Untitled"

    segment = path.split("/")[-1].replace("-", " ").replace("_", " ").strip()
    if not segment:
        return extract_domain(url) or "Untitled"
    return segment[:1].upper() + segment[1:]


def _host_regex(url: str) -> str:
    host = (urlparse(url or "").netloc or "").strip().lower()
    if not host:
        return ""
    return f"^{re.escape(host)}$"


def _extract_urls_from_text(text: str, limit: int = 6) -> list[str]:
    found: list[str] = []
    for match in _URL_RE.findall(text or ""):
        clean = match.rstrip(".,;")
        if clean and clean not in found:
            found.append(clean)
        if len(found) >= limit:
            break
    return found


def tavily_research_supported() -> bool:
    return _TAVILY_RESEARCH_AVAILABLE


def tavily_research_disabled_reason() -> str:
    return _TAVILY_RESEARCH_DISABLED_REASON


def _mark_tavily_research_unavailable(reason: str) -> None:
    global _TAVILY_RESEARCH_AVAILABLE, _TAVILY_RESEARCH_DISABLED_REASON
    _TAVILY_RESEARCH_AVAILABLE = False
    _TAVILY_RESEARCH_DISABLED_REASON = reason.strip()


def _response_error_message(resp: requests.Response) -> str:
    try:
        payload = resp.json() or {}
    except Exception:
        payload = {}

    detail = payload.get("detail")
    if isinstance(detail, dict):
        nested = detail.get("error")
        if nested:
            return str(nested).strip()
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return (resp.text or "").strip()


def _clean_site_instructions(text: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    clean = re.sub(r"^(map|crawl)\b", "", clean, flags=re.IGNORECASE).strip(" ,.;:-")
    clean = re.sub(r"^(the\s+)?(site|domain)\b", "", clean, flags=re.IGNORECASE).strip(" ,.;:-")
    clean = re.sub(r"^(and|to)\b", "", clean, flags=re.IGNORECASE).strip(" ,.;:-")
    clean = re.sub(r"\bsite:[^\s]+\b", "", clean, flags=re.IGNORECASE).strip(" ,.;:-")
    return clean


def _is_generic_result(item: ResearchItem) -> bool:
    haystack = f"{item.title} {item.snippet}".strip().lower()
    if not haystack:
        return True
    return any(hint in haystack for hint in _GENERIC_PAGE_HINTS)


class TavilyProvider(ResearchProvider):
    name = "tavily"

    def _search(
        self,
        query: str,
        max_results: int,
        deep: bool,
        *,
        include_domains: list[str] | None = None,
    ) -> list[ResearchItem]:
        body = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max(1, min(max_results, 15)),
            "search_depth": "advanced" if deep else "basic",
            "include_answer": False,
            "include_raw_content": bool(deep),
        }
        if include_domains:
            body["include_domains"] = include_domains[:20]
        resp = requests.post(TAVILY_SEARCH_URL, json=body, timeout=18)
        resp.raise_for_status()
        data = resp.json() or {}
        rows = data.get("results") or []
        out: list[ResearchItem] = []
        for row in rows[:max_results]:
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            snippet = str(row.get("content") or row.get("snippet") or "").strip()
            out.append(
                ResearchItem(
                    title=str(row.get("title") or "Untitled"),
                    url=url,
                    snippet=snippet,
                    published_at=str(row.get("published_date") or "") or None,
                )
            )
        return out

    def _extract(self, query: str, max_results: int) -> list[ResearchItem]:
        urls = _extract_urls_from_text(query, limit=min(max_results, 6))
        seed: list[ResearchItem] = []

        if not urls:
            # Extract is URL-based. Fall back to light search to discover candidate URLs.
            seed = self._search(query, max_results=max_results, deep=False)
            urls = [row.url for row in seed[: min(len(seed), 4)] if row.url]
            if not urls:
                return seed

        body = {
            "api_key": TAVILY_API_KEY,
            "urls": urls,
            "include_images": False,
            "extract_depth": "basic",
        }
        resp = requests.post(TAVILY_EXTRACT_URL, json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        rows = data.get("results") or []

        by_url = {row.url: row for row in seed}
        out: list[ResearchItem] = []
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            base = by_url.get(url)
            snippet = _truncate(str(row.get("content") or "").strip())
            out.append(
                ResearchItem(
                    title=(base.title if base else str(row.get("title") or _title_from_url(url))),
                    url=url,
                    snippet=snippet or (base.snippet if base else ""),
                    published_at=(base.published_at if base else None),
                )
            )
        return out or seed

    def _extract_urls(
        self,
        *,
        urls: list[str],
        max_results: int,
        query: str = "",
        extract_depth: str = "advanced",
        chunks_per_source: int | None = None,
    ) -> list[ResearchItem]:
        normalized_urls: list[str] = []
        for url in urls:
            clean = str(url or "").strip()
            if clean and clean not in normalized_urls:
                normalized_urls.append(clean)
        if not normalized_urls:
            return []

        body: dict[str, object] = {
            "api_key": TAVILY_API_KEY,
            "urls": normalized_urls[: min(max_results, 20)],
            "include_images": False,
            "extract_depth": extract_depth,
            "format": "markdown",
        }
        focused_query = _clean_site_instructions(query) if query else ""
        if focused_query:
            body["query"] = focused_query
            body["chunks_per_source"] = max(1, min(chunks_per_source or 3, 5))

        resp = requests.post(TAVILY_EXTRACT_URL, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json() or {}
        rows = data.get("results") or []

        out: list[ResearchItem] = []
        for row in rows[:max_results]:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            snippet = _truncate(
                str(
                    row.get("raw_content")
                    or row.get("content")
                    or row.get("markdown")
                    or row.get("text")
                    or ""
                ).strip()
            )
            out.append(
                ResearchItem(
                    title=str(row.get("title") or _title_from_url(url)),
                    url=url,
                    snippet=snippet,
                    published_at=str(row.get("published_date") or "") or None,
                )
            )
        return out

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Accept": "application/json",
        }

    def _parse_site_query(self, query: str) -> tuple[str, str]:
        text = (query or "").strip()
        if not text:
            return "", ""

        match = _URL_RE.search(text)
        if match:
            target = _normalize_site_target(match.group(0))
            instructions = _clean_site_instructions(
                f"{text[:match.start()].strip()} {text[match.end():].strip()}".strip()
            )
            return target, instructions

        domain_match = _DOMAIN_RE.search(text)
        if domain_match:
            target = _normalize_site_target(domain_match.group(0))
            instructions = _clean_site_instructions(
                f"{text[:domain_match.start()].strip()} {text[domain_match.end():].strip()}".strip()
            )
            return target, instructions

        return "", _clean_site_instructions(text)

    def _map(self, query: str, max_results: int) -> list[ResearchItem]:
        url, instructions = self._parse_site_query(query)
        if not url:
            raise ValueError("tavily map requires a site URL or domain in the query")

        body = {
            "url": url,
            "limit": max(1, min(max_results, 20)),
            "max_depth": 1,
            "max_breadth": max(8, min(max_results * 3, 24)),
            "allow_external": False,
        }
        host_regex = _host_regex(url)
        if host_regex:
            body["select_domains"] = [host_regex]
        if instructions:
            body["instructions"] = instructions

        resp = requests.post(TAVILY_MAP_URL, json=body, headers=self._auth_headers(), timeout=40)
        resp.raise_for_status()
        data = resp.json() or {}
        base_url = str(data.get("base_url") or url).strip()
        rows = data.get("results") or []
        mapped_urls = [str(raw_url or "").strip() for raw_url in rows if str(raw_url or "").strip()]

        extracted = self._extract_urls(
            urls=mapped_urls[: min(max_results, 6)],
            max_results=max_results,
            query=instructions,
            extract_depth="advanced",
            chunks_per_source=3,
        )
        extracted = _dedupe_items(extracted, max_results)
        non_generic_extracted = [item for item in extracted if not _is_generic_result(item)]
        if len(non_generic_extracted) >= min(3, max_results):
            return non_generic_extracted[:max_results]

        out: list[ResearchItem] = []
        for page_url in mapped_urls[:max_results]:
            if not page_url:
                continue
            snippet = f"Mapped page discovered on {extract_domain(base_url) or base_url}."
            if instructions:
                snippet = f"{snippet} Focus: {instructions}"
            out.append(
                ResearchItem(
                    title=_title_from_url(page_url),
                    url=page_url,
                    snippet=_truncate(snippet, limit=320),
                )
            )

        candidate_items = non_generic_extracted or extracted or out
        if len(candidate_items) >= min(3, max_results):
            return candidate_items[:max_results]

        domain = extract_domain(url)
        focused_query = _clean_site_instructions(instructions or query) or query
        search_seed = self._search(
            focused_query,
            max_results=max(max_results, 6),
            deep=True,
            include_domains=[domain] if domain else None,
        )
        extracted_search = self._extract_urls(
            urls=[item.url for item in search_seed[: min(len(search_seed), 6)] if item.url],
            max_results=max_results,
            query=focused_query,
            extract_depth="advanced",
            chunks_per_source=3,
        )
        fallback_items = _dedupe_items(
            [item for item in extracted_search if not _is_generic_result(item)] + extracted_search + candidate_items,
            max_results,
        )
        return fallback_items or candidate_items

    def _crawl(self, query: str, max_results: int) -> list[ResearchItem]:
        url, instructions = self._parse_site_query(query)
        if not url:
            raise ValueError("tavily crawl requires a site URL or domain in the query")

        body: dict[str, object] = {
            "url": url,
            "limit": max(1, min(max_results, 10)),
            "max_depth": 1,
            "max_breadth": max(12, min(max_results * 5, 50)),
            "allow_external": False,
            "extract_depth": "advanced" if any(token in extract_domain(url) for token in ("docs", "support")) else "basic",
            "format": "markdown",
        }
        host_regex = _host_regex(url)
        if host_regex:
            body["select_domains"] = [host_regex]
        if instructions:
            body["instructions"] = instructions
            body["chunks_per_source"] = 3

        resp = requests.post(TAVILY_CRAWL_URL, json=body, headers=self._auth_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json() or {}
        rows = data.get("results") or []

        out: list[ResearchItem] = []
        for row in rows[:max_results]:
            if not isinstance(row, dict):
                continue
            page_url = str(row.get("url") or "").strip()
            if not page_url:
                continue
            snippet = _truncate(
                str(
                    row.get("raw_content")
                    or row.get("content")
                    or row.get("markdown")
                    or row.get("text")
                    or ""
                ).strip()
            )
            out.append(
                ResearchItem(
                    title=str(row.get("title") or _title_from_url(page_url)),
                    url=page_url,
                    snippet=snippet or f"Crawled page from {extract_domain(url) or url}.",
                    published_at=str(row.get("published_date") or "") or None,
                )
            )
        return out

    def _research(self, query: str, max_results: int) -> list[ResearchItem]:
        if not tavily_research_supported():
            reason = tavily_research_disabled_reason() or "Tavily research is unavailable for the current account"
            raise RuntimeError(reason)

        create = requests.post(
            TAVILY_RESEARCH_URL,
            json={"input": query, "model": TAVILY_RESEARCH_MODEL},
            headers=self._auth_headers(),
            timeout=30,
        )
        if create.status_code == 432:
            reason = _response_error_message(create) or "Tavily research exceeded the current plan limit"
            _mark_tavily_research_unavailable(reason)
            raise RuntimeError(reason)
        create.raise_for_status()
        created = create.json() or {}
        request_id = str(created.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError(f"unexpected tavily research response: {created}")

        deadline = time.time() + TAVILY_RESEARCH_TIMEOUT_SECONDS
        payload: dict = {}
        while time.time() < deadline:
            poll = requests.get(
                f"{TAVILY_RESEARCH_URL}/{request_id}",
                headers=self._auth_headers(),
                timeout=30,
            )
            if poll.status_code == 432:
                reason = _response_error_message(poll) or "Tavily research exceeded the current plan limit"
                _mark_tavily_research_unavailable(reason)
                raise RuntimeError(reason)
            poll.raise_for_status()
            payload = poll.json() or {}
            status = str(payload.get("status") or "").strip().lower()
            if status == "completed":
                break
            if status in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"tavily research failed: {payload}")
            time.sleep(TAVILY_RESEARCH_POLL_SECONDS)
        else:
            raise RuntimeError(f"tavily research timed out waiting for {request_id}")

        rows = payload.get("sources") or []
        out: list[ResearchItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            snippet = _truncate(
                str(
                    row.get("content")
                    or row.get("snippet")
                    or row.get("raw_content")
                    or payload.get("content")
                    or ""
                ).strip()
            )
            out.append(
                ResearchItem(
                    title=str(row.get("title") or _title_from_url(url)),
                    url=url,
                    snippet=snippet,
                    published_at=str(row.get("published_date") or "") or None,
                )
            )

        if out:
            return out[:max_results]

        report_text = _truncate(str(payload.get("content") or "").strip(), limit=700)
        fallback_urls = _extract_urls_from_text(report_text, limit=max_results)
        return [
            ResearchItem(
                title=_title_from_url(url),
                url=url,
                snippet=report_text,
            )
            for url in fallback_urls
        ]

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

        if not TAVILY_API_KEY:
            result.errors.append("TAVILY_API_KEY not configured")
            return result

        try:
            if tool == "extract":
                result.results = self._extract(query, max_results)
            elif tool == "map":
                result.results = self._map(query, max_results)
            elif tool == "crawl":
                result.results = self._crawl(query, max_results)
            elif tool == "research":
                result.results = self._research(query, max_results)
            else:
                result.results = self._search(query, max_results=max_results, deep=(tool == "search_deep"))
        except Exception as exc:
            result.errors.append(f"tavily {tool} failed: {exc}")

        result.latency_ms = int((time.time() - started) * 1000)
        return result

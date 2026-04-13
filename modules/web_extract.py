"""Robust web article extractor used by the NAS summarization pipeline.

The design favours static HTML extraction with graceful degradation and keeps
all optional dependencies behind availability checks.  The exported
``extract_webpage`` function returns a ``WebPageContent`` dataclass that
contains the cleaned article text plus lightweight metadata required by the
downstream summarizer and Postgres writer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
import os
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote, quote

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # Optional: readability-lxml
    from readability import Document  # type: ignore

    READABILITY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    READABILITY_AVAILABLE = False

try:  # Optional: trafilatura
    from trafilatura import extract as trafilatura_extract  # type: ignore

    TRAFILATURA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRAFILATURA_AVAILABLE = False

try:  # Optional: Playwright (dynamic rendering)
    from playwright.sync_api import sync_playwright  # type: ignore

    PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 (YTV2-NAS WebExtractor)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TEXT_LENGTH_THRESHOLD = 600  # characters – avoid returning navigation junk
MAX_BODY_CHARACTERS = 120_000  # guardrail to avoid bringing down massive pages
REQUEST_TIMEOUT = 10  # seconds
MAX_RESPONSE_BYTES = 6 * 1024 * 1024  # 6 MB
MAX_REDIRECTS = 5

NAVIGATION_PHRASES = {
    "edit",
    "view history",
    "navigation",
    "tools",
    "talk",
    "read",
    "contents",
    "main page",
    "special pages",
    "print/export",
    "related changes",
    "what links here",
    "upload file",
    "permanent link",
}


@dataclass
class WebPageContent:
    """Normalized article payload returned by ``extract_webpage``."""

    source_url: str
    canonical_url: str
    title: str
    text: str
    language: Optional[str] = None
    site_name: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    top_image: Optional[str] = None
    html: Optional[str] = None
    extractor_notes: Dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable identifier used by downstream systems."""

        key = self.canonical_url or self.source_url
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return f"web:{digest}"


class WebPageExtractor:
    """Layered web extractor with graceful fallbacks."""

    def __init__(self, allow_dynamic: bool = False, session: Optional[requests.Session] = None):
        self.allow_dynamic = allow_dynamic and PLAYWRIGHT_AVAILABLE
        self.session = session or self._create_session()
        self.session.headers.update(DEFAULT_HEADERS)
        logger.info(
            "WebExtractor init: allow_dynamic=%s (playwright_available=%s)",
            bool(self.allow_dynamic),
            bool(PLAYWRIGHT_AVAILABLE),
        )

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=1,
            backoff_factor=0.6,
            status_forcelist=(500, 502, 503, 504, 429),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.max_redirects = MAX_REDIRECTS
        return session

    def _url_context_mode(self) -> str:
        raw = (os.getenv("WEB_URL_CONTEXT_MODE") or "off").strip()
        candidates = [part.strip().lower() for part in raw.split(",") if part.strip()]
        if not candidates:
            candidates = ["off"]
        for cand in candidates:
            if cand in {"0", "false", "no", "disabled", "off"}:
                return "off"
            if cand in {"auto", "always"}:
                return cand
        return "off"

    def _gemini_api_key(self) -> Optional[str]:
        # Prefer GEMINI_API_KEY (as documented by ai.google.dev examples), but
        # support GOOGLE_API_KEY as a fallback for users already using that name.
        key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if key:
            return key
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        return key or None

    def _url_context_model(self) -> str:
        raw = (os.getenv("WEB_URL_CONTEXT_MODEL") or "gemini-2.5-flash").strip()
        candidates = [part.strip() for part in raw.split(",") if part.strip()]
        return candidates[0] if candidates else "gemini-2.5-flash"

    @staticmethod
    def _gemini_model_pricing_usd_per_1m_tokens(model: str) -> Optional[Tuple[float, float]]:
        """Return (input_usd_per_1m, output_usd_per_1m) for Gemini Developer API models.

        This is best-effort, based on https://ai.google.dev/gemini-api/docs/pricing.
        """

        normalized = (model or "").strip().lower()
        if normalized.startswith("gemini-2.5-flash-lite"):
            return 0.10, 0.40
        if normalized.startswith("gemini-2.5-flash"):
            return 0.30, 2.50
        if normalized.startswith("gemini-3-flash-preview"):
            return 0.50, 3.00
        return None

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return None
                return int(float(s))
            return None
        except Exception:
            return None

    def _url_context_timeout(self) -> float:
        try:
            return float(os.getenv("WEB_URL_CONTEXT_TIMEOUT", "20"))
        except Exception:
            return 20.0

    @staticmethod
    def _looks_like_pdf_url(url: str) -> bool:
        try:
            if not url:
                return False
            lowered = url.lower()
            if lowered.endswith(".pdf"):
                return True
            parsed = urlparse(lowered)
            if parsed.path.endswith(".pdf"):
                return True
            # Common patterns for PDF endpoints
            if "/pdf/" in parsed.path:
                return True
            if "pdf" in parsed.path and ("showpdf" in parsed.path or parsed.path.endswith("/pdf")):
                return True
            return False
        except Exception:
            return False

    def _url_context_timeout_for_url(self, url: str) -> float:
        base = self._url_context_timeout()
        if not self._looks_like_pdf_url(url):
            return base
        raw = (os.getenv("WEB_URL_CONTEXT_PDF_TIMEOUT") or "").strip()
        if raw:
            try:
                return float(raw)
            except Exception:
                return base
        # PDFs frequently take longer than HTML pages to process via URL-context.
        return max(base, 60.0)

    @staticmethod
    def _rewrite_pmc_pdf_url_to_html(url: str) -> str:
        """Rewrite PMC PDF URLs to their HTML article page.

        Example:
          https://pmc.ncbi.nlm.nih.gov/articles/PMC9331845/pdf/jpm-12-01194.pdf
          -> https://pmc.ncbi.nlm.nih.gov/articles/PMC9331845/
        """

        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if host != "pmc.ncbi.nlm.nih.gov":
                return url
            path = parsed.path or ""
            match = re.match(r"^/articles/(PMC\d+)/pdf/[^/]+\.pdf$", path, flags=re.IGNORECASE)
            if not match:
                return url
            pmc_id = match.group(1)
            return f"{parsed.scheme}://{parsed.netloc}/articles/{pmc_id}/"
        except Exception:
            return url

    def _url_context_max_chars(self) -> int:
        try:
            return int(os.getenv("WEB_URL_CONTEXT_MAX_CHARS", "40000"))
        except Exception:
            return 40000

    def _extract_via_gemini_url_context(
        self,
        url: str,
        notes: Dict[str, str],
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        api_key = self._gemini_api_key()
        if not api_key:
            notes["url_context"] = "skipped (no GEMINI_API_KEY/GOOGLE_API_KEY)"
            return None

        model = self._url_context_model()
        notes["url_context_model"] = model
        timeout = self._url_context_timeout_for_url(url)
        notes["url_context_timeout_s"] = str(timeout)
        max_chars = self._url_context_max_chars()

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        prompt = (
            "You are a web article extraction assistant.\n"
            "Using the content at the URL, extract the main readable content for downstream summarization.\n\n"
            "Return JSON only (no markdown, no code fences) with keys:\n"
            "  title: string|null\n"
            "  author: string|null\n"
            "  published_at: string|null\n"
            "  text: string (plain text, paragraphs separated by \\n\\n)\n\n"
            f"Rules:\n- text must exclude navigation/ads/comments.\n- text must be at most {max_chars} characters; truncate if needed.\n\n"
            f"URL: {url}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"url_context": {}}],
        }

        def _retry_timeout_seconds(initial_timeout: float) -> Optional[float]:
            try:
                enabled = str(os.getenv("WEB_URL_CONTEXT_RETRY_ON_TIMEOUT", "1")).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                enabled = True
            if not enabled:
                return None
            try:
                retry_timeout = float(os.getenv("WEB_URL_CONTEXT_RETRY_TIMEOUT", "60") or "60")
            except Exception:
                retry_timeout = 60.0
            # Only retry if we'd actually increase the timeout.
            return retry_timeout if retry_timeout > float(initial_timeout) else None

        def _post(timeout_s: float):
            return self.session.post(
                endpoint,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout_s,
            )

        try:
            resp = _post(timeout)
        except Exception as exc:
            notes["url_context"] = f"error ({type(exc).__name__})"
            detail = str(exc).strip()
            if detail:
                notes["url_context_detail"] = detail[:180]
            # Common failure: ReadTimeout on slow/JS-heavy pages. Retry once with a higher timeout.
            if type(exc).__name__ == "ReadTimeout":
                retry_timeout = _retry_timeout_seconds(timeout)
                if retry_timeout:
                    notes["url_context_retry_timeout_s"] = str(int(retry_timeout) if retry_timeout.is_integer() else retry_timeout)
                    try:
                        resp = _post(retry_timeout)
                        notes["url_context_retried"] = "1"
                    except Exception as exc2:
                        notes["url_context_retry_error"] = f"{type(exc2).__name__}"
                        detail2 = str(exc2).strip()
                        if detail2:
                            notes["url_context_retry_detail"] = detail2[:180]
                        return None
                else:
                    return None
            else:
                return None

        if resp.status_code != 200:
            snippet = (resp.text or "").strip().replace("\n", " ")
            notes["url_context"] = f"bad_status ({resp.status_code})"
            if snippet:
                notes["url_context_detail"] = snippet[:180]
            return None

        try:
            data = resp.json() or {}
        except Exception as exc:
            notes["url_context"] = f"bad_json ({type(exc).__name__})"
            return None

        # Usage metadata (if present). We store numbers as strings to keep
        # extractor_notes JSON-friendly and consistent with other note values.
        usage = data.get("usageMetadata") or data.get("usage_metadata") or {}
        if isinstance(usage, dict):
            prompt_tokens = self._safe_int(
                usage.get("promptTokenCount") if "promptTokenCount" in usage else usage.get("prompt_token_count")
            )
            tool_tokens = self._safe_int(
                usage.get("toolUsePromptTokenCount")
                if "toolUsePromptTokenCount" in usage
                else usage.get("tool_use_prompt_token_count")
            )
            candidates_tokens = self._safe_int(
                usage.get("candidatesTokenCount")
                if "candidatesTokenCount" in usage
                else usage.get("candidates_token_count")
            )
            thoughts_tokens = self._safe_int(
                usage.get("thoughtsTokenCount") if "thoughtsTokenCount" in usage else usage.get("thoughts_token_count")
            )
            total_tokens = self._safe_int(
                usage.get("totalTokenCount") if "totalTokenCount" in usage else usage.get("total_token_count")
            )

            if prompt_tokens is not None:
                notes["url_context_prompt_tokens"] = str(prompt_tokens)
            if tool_tokens is not None:
                notes["url_context_tool_tokens"] = str(tool_tokens)
            if candidates_tokens is not None:
                notes["url_context_output_tokens"] = str(candidates_tokens)
            if thoughts_tokens is not None:
                notes["url_context_thoughts_tokens"] = str(thoughts_tokens)
            if total_tokens is not None:
                notes["url_context_total_tokens"] = str(total_tokens)

            billed_input_tokens = None
            if prompt_tokens is not None and tool_tokens is not None:
                billed_input_tokens = prompt_tokens + tool_tokens
            elif prompt_tokens is not None:
                billed_input_tokens = prompt_tokens

            billed_output_tokens = None
            if candidates_tokens is not None and thoughts_tokens is not None:
                billed_output_tokens = candidates_tokens + thoughts_tokens
            elif candidates_tokens is not None:
                billed_output_tokens = candidates_tokens
            elif thoughts_tokens is not None:
                billed_output_tokens = thoughts_tokens

            if billed_input_tokens is not None:
                notes["url_context_billed_input_tokens"] = str(billed_input_tokens)
            if billed_output_tokens is not None:
                notes["url_context_billed_output_tokens"] = str(billed_output_tokens)

            pricing = self._gemini_model_pricing_usd_per_1m_tokens(model)
            if pricing and billed_input_tokens is not None and billed_output_tokens is not None:
                in_usd_per_1m, out_usd_per_1m = pricing
                notes["url_context_pricing_input_usd_per_1m"] = str(in_usd_per_1m)
                notes["url_context_pricing_output_usd_per_1m"] = str(out_usd_per_1m)
                est_cost = (billed_input_tokens * in_usd_per_1m + billed_output_tokens * out_usd_per_1m) / 1_000_000.0
                notes["url_context_est_cost_usd"] = f"{est_cost:.8f}".rstrip("0").rstrip(".")
                if est_cost > 0:
                    est_calls_per_usd = int(round(1.0 / est_cost))
                    notes["url_context_est_calls_per_usd"] = str(est_calls_per_usd)

        raw_text = ""
        try:
            candidates = data.get("candidates") or []
            if candidates and isinstance(candidates, list):
                content = (candidates[0] or {}).get("content") or {}
                parts = content.get("parts") or []
                buf = []
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        buf.append(part["text"])
                raw_text = "\n".join(buf).strip()
        except Exception:
            raw_text = ""

        if not raw_text:
            notes["url_context"] = "empty"
            return None

        meta: Dict[str, str] = {}
        extracted_text: str = raw_text
        try:
            try_text = raw_text
            if "{" in try_text and "}" in try_text:
                try_text = try_text[try_text.find("{") : try_text.rfind("}") + 1]
            obj = json.loads(try_text)
            if isinstance(obj, dict):
                title = obj.get("title")
                author = obj.get("author")
                published = obj.get("published_at")
                text = obj.get("text")
                if isinstance(title, str) and title.strip():
                    meta["title"] = title.strip()
                if isinstance(author, str) and author.strip():
                    meta["author"] = author.strip()
                if isinstance(published, str) and published.strip():
                    meta["published"] = published.strip()
                if isinstance(text, str) and text.strip():
                    extracted_text = text.strip()
        except Exception:
            extracted_text = raw_text

        extracted_text = extracted_text.strip()
        if max_chars and len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars]

        notes["url_context"] = f"ok ({len(extracted_text)} chars)"
        return extracted_text, meta

    # -- Public API ----------------------------------------------------- #
    def extract(self, url: str) -> WebPageContent:
        """Fetch, clean, and return article content for ``url``."""

        fetch_start = time.time()
        logger.info("WebExtractor: fetching %s", url)

        url_context_mode = self._url_context_mode()

        # Initialize notes early so we can fall back to URL-context even when the
        # local HTTP fetch fails (e.g., 403 on PDFs or aggressive bot blocking).
        notes: Dict[str, str] = {
            "initial_method": "static",
            "final_url": url,
            "fetch_elapsed_ms": "0",
        }

        requested_url = url
        work_url = self._rewrite_pmc_pdf_url_to_html(url)
        if work_url != url:
            notes["pmc_rewrite"] = f"{url} -> {work_url}"
            logger.info("PMC PDF URL rewritten to HTML: %s -> %s", url, work_url)

        # If it looks like a PDF, prefer URL-context first (avoids downloading large
        # binaries and avoids some anti-bot HTML interstitials).
        if self._looks_like_pdf_url(work_url) and url_context_mode != "off":
            logger.info(
                "PDF URL detected; attempting Gemini URL context first for %s (mode=%s)",
                work_url,
                url_context_mode,
            )
            url_context_start = time.time()
            result = self._extract_via_gemini_url_context(work_url, notes)
            notes["url_context_elapsed_ms"] = str(int((time.time() - url_context_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["url_context_quality"] = cand_quality
                if cand_quality != "too_short":
                    cleaned = self._clean_text(cand_text)
                    if len(cleaned) > MAX_BODY_CHARACTERS:
                        logger.info("Truncating article text at %s characters", MAX_BODY_CHARACTERS)
                        cleaned = cleaned[:MAX_BODY_CHARACTERS]

                    final_meta = {**cand_meta}
                    notes["final_method"] = "url_context"
                    notes["final_text_chars"] = str(len(cleaned))
                    notes["final_quality"] = cand_quality
                    return WebPageContent(
                        source_url=requested_url,
                        canonical_url=final_meta.get("canonical_url") or work_url,
                        title=final_meta.get("title") or work_url,
                        text=cleaned,
                        language=final_meta.get("language"),
                        site_name=final_meta.get("site_name"),
                        author=final_meta.get("author"),
                        published_at=final_meta.get("published"),
                        top_image=final_meta.get("top_image"),
                        html=None,
                        extractor_notes=notes,
                    )

        # Wikipedia fast-path via official REST APIs (opt-in)
        with contextlib.suppress(Exception):
            wiki_mode = (os.getenv("WIKI_API_MODE") or "auto").strip().lower()
            if wiki_mode not in {"auto", "full", "summary", "off", "disabled"}:
                wiki_mode = "auto"
            if wiki_mode not in {"off", "disabled"}:
                wiki_content = self._maybe_extract_wikipedia(url, mode=wiki_mode)
                if wiki_content:
                    return wiki_content

        try:
            response, final_url = self._fetch(work_url)
        except Exception as exc:
            notes["fetch_elapsed_ms"] = f"{int((time.time() - fetch_start) * 1000)}"
            notes["fetch_error"] = type(exc).__name__
            if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                with contextlib.suppress(Exception):
                    notes["fetch_status"] = str(exc.response.status_code)
                with contextlib.suppress(Exception):
                    notes["final_url"] = str(exc.response.url or url)
                with contextlib.suppress(Exception):
                    cf_mitigated = (exc.response.headers.get("cf-mitigated") or "").strip()
                    if cf_mitigated:
                        notes["cf_mitigated"] = cf_mitigated
                with contextlib.suppress(Exception):
                    server = (exc.response.headers.get("Server") or "").strip()
                    if server:
                        notes["server"] = server

            if url_context_mode != "off":
                logger.info(
                    "Local fetch failed; attempting Gemini URL context fallback for %s (mode=%s)",
                    work_url,
                    url_context_mode,
                )
                url_context_start = time.time()
                result = self._extract_via_gemini_url_context(work_url, notes)
                notes["url_context_elapsed_ms"] = str(int((time.time() - url_context_start) * 1000))
                if result:
                    cand_text, cand_meta = result
                    cand_quality = self._assess_quality(cand_text)
                    notes["url_context_quality"] = cand_quality

                    cleaned = self._clean_text(cand_text)
                    if len(cleaned) > MAX_BODY_CHARACTERS:
                        logger.info("Truncating article text at %s characters", MAX_BODY_CHARACTERS)
                        cleaned = cleaned[:MAX_BODY_CHARACTERS]

                    final_meta = {**cand_meta}
                    notes["final_method"] = "url_context"
                    notes["final_text_chars"] = str(len(cleaned))
                    notes["final_quality"] = cand_quality
                    return WebPageContent(
                        source_url=requested_url,
                        canonical_url=final_meta.get("canonical_url") or work_url,
                        title=final_meta.get("title") or work_url,
                        text=cleaned,
                        language=final_meta.get("language"),
                        site_name=final_meta.get("site_name"),
                        author=final_meta.get("author"),
                        published_at=final_meta.get("published"),
                        top_image=final_meta.get("top_image"),
                        html=None,
                        extractor_notes=notes,
                    )

            # If we attempted URL context and it failed, raise a clearer error than the raw 403/blocked response.
            url_context_note = (notes.get("url_context") or "").strip()
            if url_context_mode != "off" and url_context_note:
                fetch_status = (notes.get("fetch_status") or "").strip()
                cf_hint = ""
                if (notes.get("cf_mitigated") or "").strip():
                    cf_hint = " (Cloudflare challenge)"
                detail = (notes.get("url_context_detail") or "").strip()
                detail = (detail[:140] + "…") if len(detail) > 140 else detail
                msg = f"Blocked fetching URL{cf_hint}"
                if fetch_status:
                    msg += f" (HTTP {fetch_status})"
                msg += f"; Gemini URL context fallback: {url_context_note}"
                if detail:
                    msg += f" ({detail})"
                msg += "."
                raise RuntimeError(msg) from exc

            raise

        logger.debug(
            "WebExtractor: fetch done status=%s len=%s final_url=%s",
            getattr(response, "status_code", "-"),
            len(getattr(response, "content", b"")),
            final_url,
        )
        notes["final_url"] = final_url
        notes["fetch_elapsed_ms"] = f"{int((time.time() - fetch_start) * 1000)}"

        # If the fetch returned a non-HTML document (e.g., PDF), prefer URL-context.
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and url_context_mode != "off":
            logger.info(
                "Non-HTML content-type detected; attempting Gemini URL context fallback for %s (content-type=%s mode=%s)",
                final_url,
                content_type or "unknown",
                url_context_mode,
            )
            url_context_start = time.time()
            result = self._extract_via_gemini_url_context(final_url, notes)
            notes["url_context_elapsed_ms"] = str(int((time.time() - url_context_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["url_context_quality"] = cand_quality
                cleaned = self._clean_text(cand_text)
                if len(cleaned) > MAX_BODY_CHARACTERS:
                    logger.info("Truncating article text at %s characters", MAX_BODY_CHARACTERS)
                    cleaned = cleaned[:MAX_BODY_CHARACTERS]

                final_meta = {**cand_meta}
                notes["final_method"] = "url_context"
                notes["final_text_chars"] = str(len(cleaned))
                notes["final_quality"] = cand_quality
                return WebPageContent(
                    source_url=requested_url,
                    canonical_url=final_meta.get("canonical_url") or final_url,
                    title=final_meta.get("title") or final_url,
                    text=cleaned,
                    language=final_meta.get("language"),
                    site_name=final_meta.get("site_name"),
                    author=final_meta.get("author"),
                    published_at=final_meta.get("published"),
                    top_image=final_meta.get("top_image"),
                    html=None,
                    extractor_notes=notes,
                )

            # Avoid attempting HTML parsers on binary documents (can yield garbage).
            url_context_note = (notes.get("url_context") or "").strip()
            if url_context_note:
                raise RuntimeError(
                    f"Non-HTML content-type ({content_type}) for {final_url}; Gemini URL context fallback: {url_context_note}."
                )
            raise ValueError(f"Non-HTML content-type ({content_type}) for {final_url}")

        prepare_start = time.time()
        html, metadata = self._prepare_html(response)
        prepare_elapsed_ms = int((time.time() - prepare_start) * 1000)

        static_start = time.time()
        text, meta = self._extract_static(html, final_url, metadata)
        static_elapsed_ms = int((time.time() - static_start) * 1000)
        notes["prepare_elapsed_ms"] = str(prepare_elapsed_ms)
        notes["static_elapsed_ms"] = str(static_elapsed_ms)

        best_text = text
        best_meta = meta
        best_method = "static"
        best_quality = self._assess_quality(text)
        notes["static_quality"] = best_quality

        if best_quality != "good":
            logger.debug(
                "Static extraction quality=%s (%s chars) for %s",
                best_quality,
                len(text),
                final_url,
            )
            readability_start = time.time()
            result = self._extract_via_readability(html, final_url, metadata, notes)
            notes["readability_elapsed_ms"] = str(int((time.time() - readability_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["readability_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality
                    best_method = "readability"

        if best_quality != "good":
            logger.debug("Attempting trafilatura fallback for %s", final_url)
            trafilatura_start = time.time()
            result = self._extract_via_trafilatura(html, final_url, metadata, notes)
            notes["trafilatura_elapsed_ms"] = str(int((time.time() - trafilatura_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["trafilatura_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality
                    best_method = "trafilatura"

        if best_quality != "good" and self.allow_dynamic:
            logger.info(
                "Attempting dynamic rendering for %s (playwright_available=%s)",
                final_url,
                bool(PLAYWRIGHT_AVAILABLE),
            )
            playwright_start = time.time()
            result = self._extract_via_playwright(final_url, notes)
            notes["playwright_elapsed_ms"] = str(int((time.time() - playwright_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["playwright_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality
                    best_method = "playwright"

        if url_context_mode == "always" or (url_context_mode == "auto" and best_quality != "good"):
            logger.info("Attempting Gemini URL context fallback for %s (mode=%s)", final_url, url_context_mode)
            url_context_start = time.time()
            result = self._extract_via_gemini_url_context(final_url, notes)
            notes["url_context_elapsed_ms"] = str(int((time.time() - url_context_start) * 1000))
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["url_context_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality
                    best_method = "url_context"

        if best_quality != "good":
            logger.warning("All extraction methods yielded suboptimal content (%s) for %s", best_quality, final_url)

        cleaned = self._clean_text(best_text)
        if len(cleaned) > MAX_BODY_CHARACTERS:
            logger.info("Truncating article text at %s characters", MAX_BODY_CHARACTERS)
            cleaned = cleaned[:MAX_BODY_CHARACTERS]

        final_meta = {**metadata, **best_meta}
        notes["final_method"] = best_method
        notes["final_text_chars"] = str(len(cleaned))
        notes["final_quality"] = best_quality

        return WebPageContent(
            source_url=requested_url,
            canonical_url=final_meta.get("canonical_url") or final_url,
            title=final_meta.get("title") or final_meta.get("og:title") or final_url,
            text=cleaned,
            language=final_meta.get("language"),
            site_name=final_meta.get("site_name"),
            author=final_meta.get("author"),
            published_at=final_meta.get("published"),
            top_image=final_meta.get("top_image"),
            html=html,
            extractor_notes=notes,
        )

    # -- Wikipedia fast-path (REST APIs) ------------------------------- #
    def _maybe_extract_wikipedia(self, url: str, *, mode: str = "auto") -> Optional[WebPageContent]:
        """Use official Wikipedia APIs to retrieve sanitized content.

        Modes:
          - summary: REST page/summary JSON only
          - full: REST page/mobile-html for full, sectioned HTML
          - auto: summary first; if long/standard article then mobile-html
        """
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if not host.endswith("wikipedia.org"):
            return None

        # Determine language and title from URL
        lang = host.split(".")[0] if host.count(".") >= 2 else "en"
        title = None
        if parsed.path.startswith("/wiki/"):
            title = parsed.path[len("/wiki/"):]
        elif parsed.path.startswith("/w/") and parsed.query:
            # e.g., /w/index.php?title=...
            with contextlib.suppress(Exception):
                from urllib.parse import parse_qs
                qs = parse_qs(parsed.query)
                if "title" in qs and qs["title"]:
                    title = qs["title"][0]
        if not title:
            title = parsed.path.lstrip("/") or ""
        title = unquote(title or "")
        if not title:
            return None

        safe_title = quote(title.replace(" ", "_"))
        api_base = f"https://{lang}.wikipedia.org/api/rest_v1/page"
        try:
            cutoff = int(os.getenv("WIKI_SUMMARY_CHAR_CUTOFF", "1500"))
        except Exception:
            cutoff = 1500

        def _http_get_json(path: str) -> Optional[dict]:
            r = self.session.get(
                f"{api_base}/{path}",
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

        def _http_get_text(path: str) -> Optional[str]:
            r = self.session.get(
                f"{api_base}/{path}",
                headers={"Accept": "text/html"},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text

        summary: Optional[dict] = None
        if mode in {"auto", "summary"}:
            summary = _http_get_json(f"summary/{safe_title}")
            if not summary and mode == "summary":
                return None

        def _build_from_summary(data: dict) -> WebPageContent:
            extract = (data.get("extract") or "").strip()
            extract_html = (data.get("extract_html") or "").strip()
            text_plain = extract
            if not text_plain and extract_html:
                soup = BeautifulSoup(extract_html, "html.parser")
                text_plain = soup.get_text("\n", strip=True)
            canon = (
                ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
                or data.get("canonical")
                or url
            )
            title_str = (data.get("title") or title).strip()
            html_min = extract_html or "\n".join(
                f"<p>{p}</p>" for p in text_plain.split("\n") if p.strip()
            )
            return WebPageContent(
                source_url=url,
                canonical_url=canon,
                title=title_str,
                text=text_plain,
                language=(data.get("lang") or lang),
                site_name="Wikipedia",
                author=None,
                published_at=None,
                top_image=(data.get("thumbnail") or {}).get("source"),
                html=html_min,
                extractor_notes={
                    "initial_method": "wikipedia-rest-summary",
                    "final_quality": "good" if len(text_plain) >= 200 else "short",
                },
            )

        if summary:
            s_type = (summary.get("type") or "").lower()
            extract_len = len((summary.get("extract") or "").strip())
            if mode == "summary" or s_type != "standard" or extract_len < cutoff:
                return _build_from_summary(summary)

        if mode in {"auto", "full"}:
            html_text = _http_get_text(f"mobile-html/{safe_title}")
            if html_text:
                soup = BeautifulSoup(html_text, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "aside"]):
                    tag.decompose()
                article = soup
                text = article.get_text("\n", strip=True)
                canon = summary and (
                    ((summary.get("content_urls") or {}).get("desktop") or {}).get("page")
                )
                canon = canon or url
                title_str = (summary.get("title") if summary else None) or title
                return WebPageContent(
                    source_url=url,
                    canonical_url=canon,
                    title=title_str,
                    text=self._clean_text(text),
                    language=(summary.get("lang") if summary else None) or lang,
                    site_name="Wikipedia",
                    author=None,
                    published_at=None,
                    top_image=((summary.get("thumbnail") or {}).get("source") if summary else None),
                    html=html_text,
                    extractor_notes={
                        "initial_method": "wikipedia-rest-mobile-html",
                        "final_quality": self._assess_quality(text),
                    },
                )

        return None

    # -- Fetch helpers -------------------------------------------------- #
    def _fetch(self, url: str) -> Tuple[requests.Response, str]:
        """Fetch URL with fallbacks for sites that block non-browser clients.

        Steps:
        - Try default request
        - On 403: retry with Safari UA + referer hint (WEB_EXTRACT_REFERER or Flipboard)
        - On 403 again: strip utm_* and retry once
        Logs each step for diagnostics.
        """
        def _do_get(target: str, extra: Optional[Dict[str, str]] = None) -> requests.Response:
            headers = dict(self.session.headers)
            if extra:
                headers.update(extra)
            logger.debug("WebExtractor GET %s headers=%s", target, {k: headers[k] for k in ['User-Agent','Referer'] if k in headers})
            return self.session.get(target, timeout=REQUEST_TIMEOUT, allow_redirects=True, headers=headers)

        try:
            resp = _do_get(url)
        except requests.RequestException as exc:
            logger.error("HTTP fetch failed for %s: %s", url, exc)
            raise

        final_url = resp.url
        if resp.status_code == 403:
            logger.info("WebExtractor: 403 on first attempt for %s", final_url)
            safari_ua = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            )
            referer = os.getenv("WEB_EXTRACT_REFERER", "https://flipboard.com/")
            try:
                resp = _do_get(final_url, {
                    "User-Agent": safari_ua,
                    "Referer": referer,
                    "Accept-Encoding": "gzip, deflate, br",
                })
                final_url = resp.url
            except requests.RequestException:
                pass
            if resp.status_code == 403:
                # Strip tracking params (utm_*) and retry once more
                try:
                    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
                    parts = list(urlsplit(final_url))
                    q = [(k, v) for k, v in parse_qsl(parts[3]) if not k.lower().startswith("utm_")]
                    parts[3] = urlencode(q)
                    cleaned = urlunsplit(parts)
                    logger.info("WebExtractor: 403 persists, retrying without utm_*: %s", cleaned)
                    resp = _do_get(cleaned, {
                        "User-Agent": safari_ua,
                        "Referer": referer,
                        "Accept-Encoding": "gzip, deflate, br",
                    })
                    final_url = resp.url
                except Exception:
                    pass
            # Try AMP variant as a last resort if still blocked
            if resp.status_code == 403:
                try:
                    from urllib.parse import urlsplit, urlunsplit
                    parts = list(urlsplit(final_url))
                    if not parts[2].endswith('/amp'):
                        parts[2] = parts[2].rstrip('/') + '/amp'
                        amp_url = urlunsplit(parts)
                        logger.info("WebExtractor: 403 still present, trying AMP variant: %s", amp_url)
                        resp = _do_get(amp_url, {
                            "User-Agent": safari_ua,
                            "Referer": referer,
                            "Accept-Encoding": "gzip, deflate, br",
                        })
                        final_url = resp.url
                except Exception:
                    pass
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            logger.warning("URL %s returned non-HTML content-type: %s", final_url, content_type)
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length:
            with contextlib.suppress(ValueError):
                if int(content_length) > MAX_RESPONSE_BYTES:
                    raise ValueError(f"Response too large ({content_length} bytes) for {final_url}")

        content = resp.content
        if len(content) > MAX_RESPONSE_BYTES:
            raise ValueError(f"Downloaded content exceeds {MAX_RESPONSE_BYTES} bytes for {final_url}")

        encoding = resp.encoding
        if encoding:
            normalized = encoding.lower()
            if normalized in {"iso-8859-1", "latin-1", "ascii"} and resp.apparent_encoding:
                encoding = resp.apparent_encoding
        else:
            encoding = resp.apparent_encoding or "utf-8"
        html = content.decode(encoding, errors="replace")
        resp.encoding = encoding
        resp._content = html.encode(encoding, errors="replace")  # type: ignore[attr-defined]
        return resp, final_url

    def _prepare_html(self, response: requests.Response) -> Tuple[str, Dict[str, str]]:
        html = response.text
        metadata = {
            "title": "",
            "canonical_url": "",
            "language": "",
            "site_name": "",
            "author": "",
            "published": "",
            "top_image": "",
        }
        soup = BeautifulSoup(html, "html.parser")
        metadata.update(self._extract_meta_tags(soup, response.url))
        return html, metadata

    # -- Extraction layers ---------------------------------------------- #
    def _extract_static(self, html: str, base_url: str, metadata: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "aside"]):
            tag.decompose()

        article = self._find_main_article(soup)
        text = article.get_text("\n", strip=True) if article else soup.get_text("\n", strip=True)

        if not metadata.get("title"):
            metadata["title"] = soup.title.string.strip() if soup.title and soup.title.string else ""

        metadata.setdefault("top_image", self._find_top_image(soup, base_url))
        return text, metadata

    def _extract_via_readability(
        self, html: str, base_url: str, metadata: Dict[str, str], notes: Dict[str, str]
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        if not READABILITY_AVAILABLE:
            notes["readability"] = "skipped (not installed)"
            return None
        try:
            doc = Document(html)
            summary_html = doc.summary()
            summary_soup = BeautifulSoup(summary_html, "html.parser")
            text = summary_soup.get_text("\n", strip=True)
            metadata = {**metadata}
            metadata["title"] = metadata.get("title") or doc.short_title()
            metadata.setdefault("top_image", self._find_top_image(summary_soup, base_url))
            notes["readability"] = f"ok ({len(text)} chars)"
            return text, metadata
        except Exception as exc:  # pragma: no cover - library edge cases
            logger.debug("Readability failed for %s: %s", base_url, exc)
            notes["readability"] = f"error ({exc})"
            return None

    def _extract_via_trafilatura(
        self, html: str, base_url: str, metadata: Dict[str, str], notes: Dict[str, str]
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        if not TRAFILATURA_AVAILABLE:
            notes["trafilatura"] = "skipped (not installed)"
            return None
        try:
            text = trafilatura_extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
                url=base_url,
                output_format="txt",
            )
            if not text:
                notes["trafilatura"] = "empty"
                return None
            notes["trafilatura"] = f"ok ({len(text)} chars)"
            return text, metadata
        except Exception as exc:  # pragma: no cover
            logger.debug("Trafilatura failed for %s: %s", base_url, exc)
            notes["trafilatura"] = f"error ({exc})"
            return None

    def _extract_via_playwright(
        self, url: str, notes: Dict[str, str]
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        if not PLAYWRIGHT_AVAILABLE:
            notes["playwright"] = "skipped (not installed)"
            return None
        try:
            with sync_playwright() as pw:  # pragma: no cover - requires browser
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
                html = page.content()
                browser.close()
        except Exception as exc:
            logger.debug("Playwright failed for %s: %s", url, exc)
            notes["playwright"] = f"error ({exc})"
            return None

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        metadata = self._extract_meta_tags(soup, url)
        notes["playwright"] = f"ok ({len(text)} chars)"
        return text, metadata

    # -- Metadata helpers ------------------------------------------------ #
    def _extract_meta_tags(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        if soup.title and soup.title.string:
            meta["title"] = soup.title.string.strip()

        canonical = soup.find("link", rel=lambda value: value and "canonical" in value.lower())  # type: ignore[arg-type]
        if canonical and canonical.get("href"):
            meta["canonical_url"] = urljoin(base_url, canonical["href"].strip())
        else:
            meta["canonical_url"] = base_url

        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta["language"] = html_tag["lang"].strip().lower()

        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").lower()
            content = (tag.get("content") or "").strip()
            if not name or not content:
                continue
            if name in ("author", "article:author", "dc.creator"):
                meta.setdefault("author", content)
            elif name in ("og:site_name", "application-name"):
                meta.setdefault("site_name", content)
            elif name in ("og:title", "twitter:title") and not meta.get("title"):
                meta["title"] = content
            elif name in ("og:image", "twitter:image"):
                meta.setdefault("top_image", urljoin(base_url, content))
            elif name in ("article:published_time", "pubdate", "dc.date", "date"):
                meta.setdefault("published", content)
            elif name in ("og:locale", "content-language") and not meta.get("language"):
                meta["language"] = content.lower()

        return meta

    def _find_main_article(self, soup: BeautifulSoup) -> Optional[Tag]:
        candidates = []
        selectors = (
            "article",
            "main",
            "[role=main]",
            ".article-body",
            ".post-content",
            ".StoryBodyCompanionColumn",
        )
        for selector in selectors:
            found = soup.select_one(selector)
            if found and self._is_valid_candidate(found):
                candidates.append(found)
        if candidates:
            candidates.sort(key=lambda node: len(node.get_text()))
            return candidates[-1]
        return None

    @staticmethod
    def _is_valid_candidate(node: Tag) -> bool:
        text = node.get_text("", strip=True)
        return bool(text) and len(text) > 300

    def _find_top_image(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return urljoin(base_url, og_image["content"].strip())

        first_img = soup.find("img")
        if first_img and first_img.get("src"):
            return urljoin(base_url, first_img["src"].strip())
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _assess_quality(self, text: str) -> str:
        if not text:
            return "too_short"
        if len(text) < TEXT_LENGTH_THRESHOLD:
            return "too_short"

        sample = text[:2000].lower()
        nav_hits = sum(sample.count(token) for token in NAVIGATION_PHRASES)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        short_lines = sum(1 for line in lines if len(line.split()) <= 3)
        short_ratio = (short_lines / len(lines)) if lines else 0.0

        if nav_hits >= 5 or short_ratio > 0.4:
            return "nav_heavy"
        return "good"

    @staticmethod
    def _is_better_quality(candidate_quality: str, current_quality: str, candidate_text: str, current_text: str) -> bool:
        ranks = {"good": 3, "nav_heavy": 2, "too_short": 1, "unknown": 0}
        cand_rank = ranks.get(candidate_quality, 0)
        curr_rank = ranks.get(current_quality, 0)
        if cand_rank != curr_rank:
            return cand_rank > curr_rank
        return len(candidate_text) > len(current_text)


def extract_webpage(url: str, allow_dynamic: bool = False) -> WebPageContent:
    """Convenience wrapper used by callers outside this module."""

    extractor = WebPageExtractor(allow_dynamic=allow_dynamic)
    return extractor.extract(url)


def _demo(url: str) -> None:  # pragma: no cover - manual testing helper
    content = extract_webpage(url)
    print(f"ID: {content.id}")
    print(f"Title: {content.title}")
    print(f"Canonical URL: {content.canonical_url}")
    print(f"Language: {content.language}")
    print(f"Author: {content.author}")
    print(f"Site: {content.site_name}")
    print(f"Published: {content.published_at}")
    print(f"Top image: {content.top_image}")
    print(f"Text preview ({len(content.text)} chars):\n{content.text[:500]}...")


if __name__ == "__main__":  # pragma: no cover
    test_url = "https://example.com"
    _demo(test_url)

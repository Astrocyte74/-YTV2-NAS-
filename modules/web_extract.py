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
import logging
import re
import time
from dataclasses import dataclass, field
import os
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

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
REQUEST_TIMEOUT = 15  # seconds
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
            total=3,
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

    # -- Public API ----------------------------------------------------- #
    def extract(self, url: str) -> WebPageContent:
        """Fetch, clean, and return article content for ``url``."""

        fetch_start = time.time()
        logger.info("WebExtractor: fetching %s", url)
        response, final_url = self._fetch(url)
        logger.debug(
            "WebExtractor: fetch done status=%s len=%s final_url=%s",
            getattr(response, "status_code", "-"),
            len(getattr(response, "content", b"")),
            final_url,
        )
        html, metadata = self._prepare_html(response)
        text, meta = self._extract_static(html, final_url, metadata)

        notes: Dict[str, str] = {
            "initial_method": "static",
            "final_url": final_url,
            "fetch_elapsed_ms": f"{int((time.time() - fetch_start) * 1000)}",
        }

        best_text = text
        best_meta = meta
        best_quality = self._assess_quality(text)
        notes["static_quality"] = best_quality

        if best_quality != "good":
            logger.debug(
                "Static extraction quality=%s (%s chars) for %s",
                best_quality,
                len(text),
                final_url,
            )
            result = self._extract_via_readability(html, final_url, metadata, notes)
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["readability_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality

        if best_quality != "good":
            logger.debug("Attempting trafilatura fallback for %s", final_url)
            result = self._extract_via_trafilatura(html, final_url, metadata, notes)
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["trafilatura_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality

        if best_quality != "good" and self.allow_dynamic:
            logger.info(
                "Attempting dynamic rendering for %s (playwright_available=%s)",
                final_url,
                bool(PLAYWRIGHT_AVAILABLE),
            )
            result = self._extract_via_playwright(final_url, notes)
            if result:
                cand_text, cand_meta = result
                cand_quality = self._assess_quality(cand_text)
                notes["playwright_quality"] = cand_quality
                if self._is_better_quality(cand_quality, best_quality, cand_text, best_text):
                    best_text, best_meta, best_quality = cand_text, cand_meta, cand_quality

        if best_quality != "good":
            logger.warning("All extraction methods yielded suboptimal content (%s) for %s", best_quality, final_url)

        cleaned = self._clean_text(best_text)
        if len(cleaned) > MAX_BODY_CHARACTERS:
            logger.info("Truncating article text at %s characters", MAX_BODY_CHARACTERS)
            cleaned = cleaned[:MAX_BODY_CHARACTERS]

        final_meta = {**metadata, **best_meta}
        notes["final_quality"] = best_quality

        return WebPageContent(
            source_url=url,
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

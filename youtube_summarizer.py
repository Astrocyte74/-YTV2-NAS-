#!/usr/bin/env python3
"""
YouTube Video Summarizer using MCP-Use Library
Extracts transcripts from YouTube videos and generates intelligent summaries
"""

import asyncio
import contextlib
import os
import logging
import json
import logging
import os
import re
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import yt_dlp

# Try to import youtube-transcript-api for better transcript access
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_API_AVAILABLE = True
    
    # Test the API to ensure correct version/method exists
    import inspect
    api_instance = YouTubeTranscriptApi()
    if not hasattr(api_instance, 'fetch'):
        print("⚠️  youtube-transcript-api version incompatible, will use yt-dlp only")
        TRANSCRIPT_API_AVAILABLE = False
    else:
        print("✅ youtube-transcript-api loaded successfully")
        
except ImportError:
    print("⚠️  youtube-transcript-api not available, will use yt-dlp only")
    TRANSCRIPT_API_AVAILABLE = False
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from modules.ollama_client import OllamaClientError
from modules import ollama_client as oc
import requests

from modules.tts_hub import TTSHubClient, LocalTTSUnavailable


def _env_flag(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


SUMMARY_IMAGE_AI2_ENABLED = _env_flag("SUMMARY_IMAGE_ENABLE_AI2", "true")

# --- Transcript rate limiter / circuit breaker ---
import time

class TranscriptLimiter:
    def __init__(self, min_interval: float, threshold: int, cooldown_secs: float):
        self.min_interval = float(min_interval)
        self.threshold = int(threshold)
        self.cooldown_secs = float(cooldown_secs)
        self._next_allowed = 0.0
        self._fail_count = 0
        self._cooldown_until = 0.0

    def update(self, min_interval: float, threshold: int, cooldown_secs: float) -> None:
        self.min_interval = float(min_interval)
        self.threshold = int(threshold)
        self.cooldown_secs = float(cooldown_secs)

    def before_request(self) -> bool:
        now = time.monotonic()
        # If in cooldown, block
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            try:
                logging.info("Transcript breaker active: cooldown %.0fs left", remaining)
            except Exception:
                pass
            return False
        # Enforce min interval
        sleep_for = max(0.0, self._next_allowed - now)
        if sleep_for > 0:
            try:
                logging.info("Transcript rate limit: sleeping %.1fs", sleep_for)
                time.sleep(min(sleep_for, 5.0))
            except Exception:
                pass
        return True

    def on_success(self) -> None:
        now = time.monotonic()
        self._fail_count = 0
        self._next_allowed = now + max(0.0, self.min_interval)

    def on_429(self) -> None:
        self._fail_count += 1
        if self._fail_count >= max(1, self.threshold):
            self._cooldown_until = time.monotonic() + max(0.0, self.cooldown_secs)
            try:
                logging.warning("Transcript breaker entering cooldown for %.0fs after repeated 429s", self.cooldown_secs)
            except Exception:
                pass
            self._fail_count = 0


_TRANSCRIPT_LIMITER: TranscriptLimiter | None = None

# --- Simple in-memory transcript cache ---
from typing import Tuple
_TRANSCRIPT_CACHE: dict[str, Tuple[str, str, float]] = {}

# Optional: cache transcript segments (timestamped)
# video_id -> (segments:list[dict], ts:float)
_TRANSCRIPT_SEGMENT_CACHE: dict[str, Tuple[List[dict], float]] = {}

def _transcript_cache_ttl() -> float:
    try:
        return float(os.getenv('YT_TRANSCRIPT_CACHE_TTL', '86400') or '86400')
    except Exception:
        return 86400.0

def _transcript_cache_get(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    ttl = _transcript_cache_ttl()
    rec = _TRANSCRIPT_CACHE.get(video_id)
    if not rec:
        return None, None
    text, lang, ts = rec
    if (time.time() - ts) > ttl:
        # Expired
        try:
            del _TRANSCRIPT_CACHE[video_id]
        except Exception:
            pass
        return None, None
    try:
        logging.info("Transcript cache hit for %s (age=%.0fs)", video_id, time.time() - ts)
    except Exception:
        pass
    return text, lang

# --- Simple in-memory metadata cache ---
_METADATA_CACHE: dict[str, Tuple[dict, float]] = {}

def _metadata_cache_ttl() -> float:
    try:
        return float(os.getenv('YT_METADATA_CACHE_TTL', '86400') or '86400')
    except Exception:
        return 86400.0

def _metadata_cache_get(video_id: str) -> Optional[dict]:
    ttl = _metadata_cache_ttl()
    rec = _METADATA_CACHE.get(video_id)
    if not rec:
        return None
    data, ts = rec
    if (time.time() - ts) > ttl:
        try:
            del _METADATA_CACHE[video_id]
        except Exception:
            pass
        return None
    try:
        logging.info("Metadata cache hit for %s (age=%.0fs)", video_id, time.time() - ts)
    except Exception:
        pass
    return data

def _metadata_cache_put(video_id: str, data: dict) -> None:
    try:
        _METADATA_CACHE[video_id] = (data, time.time())
        logging.info("Metadata cached for %s", video_id)
    except Exception:
        pass
    

def _transcript_cache_put(video_id: str, text: str, lang: str) -> None:
    try:
        _TRANSCRIPT_CACHE[video_id] = (text, lang or '', time.time())
        logging.info("Transcript cached for %s (len=%d)", video_id, len(text or ''))
    except Exception:
        pass

def _transcript_segment_cache_put(video_id: str, segments: List[dict]) -> None:
    try:
        _TRANSCRIPT_SEGMENT_CACHE[video_id] = (segments or [], time.time())
        logging.info("Transcript segments cached for %s (count=%d)", video_id, len(segments or []))
    except Exception:
        pass

def _transcript_segment_cache_get(video_id: str) -> Optional[List[dict]]:
    ttl = _transcript_cache_ttl()
    rec = _TRANSCRIPT_SEGMENT_CACHE.get(video_id)
    if not rec:
        return None
    segments, ts = rec
    if (time.time() - ts) > ttl:
        try:
            del _TRANSCRIPT_SEGMENT_CACHE[video_id]
        except Exception:
            pass
        return None
    try:
        logging.info("Transcript segment cache hit for %s (age=%.0fs)", video_id, time.time() - ts)
    except Exception:
        pass
    return segments

# Import LLM configuration manager
from llm_config import llm_config

from modules.services import summary_image_service

try:
    from modules.sources import (
        RedditFetcher,
        RedditFetcherError,
        WebPageFetcher,
        WebFetcherError,
    )
except ImportError:  # pragma: no cover - optional dependency
    RedditFetcher = None
    WebPageFetcher = None

    class RedditFetcherError(Exception):
        """Placeholder error used when Reddit dependencies are unavailable."""

    class WebFetcherError(Exception):
        """Placeholder error used when web dependencies are unavailable."""

class YouTubeSummarizer:
    def __init__(self, llm_provider: str = None, model: str = None, ollama_base_url: str = None):
        """Initialize the YouTube Summarizer with mkpy LLM integration
        
        Args:
            llm_provider: Optional override for LLM provider. If None, uses mkpy configuration
            model: Optional override for model. If None, uses mkpy configuration  
            ollama_base_url: Base URL for Ollama server (default from config)
        """
        self.downloads_dir = Path("./downloads")
        self.downloads_dir.mkdir(exist_ok=True)

        # Get LLM configuration from mkpy system
        try:
            self.llm_provider, self.model, api_key = llm_config.get_model_config(llm_provider, model)
        except ValueError as e:
            print(f"🔴 {e}")
            raise

        # Set Ollama base URL; if a hub is configured, prefer its Ollama proxy
        self.ollama_base_url = ollama_base_url or None
        try:
            if self.llm_provider == "ollama" and not self.ollama_base_url:
                hub_base = os.getenv("TTSHUB_API_BASE")
                if hub_base:
                    self.ollama_base_url = hub_base.rstrip("/") + "/ollama"
                else:
                    self.ollama_base_url = llm_config.ollama_host
        except Exception:
            self.ollama_base_url = self.ollama_base_url or llm_config.ollama_host

        # Initialize LLM based on determined configuration
        self._initialize_llm(api_key)

        # Lazy-initialized Reddit fetcher
        self._reddit_fetcher: Optional["RedditFetcher"] = None
        # Lazy-initialized web page fetcher
        self._web_fetcher: Optional["WebPageFetcher"] = None
        # Lazy-initialized local TTS hub client
        self._tts_hub_client: Optional[TTSHubClient] = None

        # Optional status callback for UI progress (set by caller)
        self.status_callback = None  # type: ignore[attr-defined]

    def _notify_status(self, message: str) -> None:
        """Notify external status callback if configured (non-blocking)."""
        try:
            cb = getattr(self, "status_callback", None)
            if callable(cb) and isinstance(message, str) and message.strip():
                cb(message.strip())
        except Exception:
            pass

    def _apply_ytdlp_env(self, ydl_opts: dict) -> None:
        """Apply environment-driven tweaks to yt-dlp options.

        Supported env vars (container/Portainer configurable):
          - YTDLP_SAFE_MODE=1             → use android-only client
          - YTDLP_FORCE_CLIENT=android|web|tv|web_safari
          - YTDLP_SLEEP_REQUESTS=int      → request pacing (seconds)
          - YTDLP_RETRIES=int             → total retries (also applied to fragment_retries)
          - YTDLP_COOKIES_FILE=/path.txt  → Netscape cookie file path
          - YTDLP_FORCE_STACK=ipv4|ipv6   → select IP stack (maps to source_address)
          - YT_DLP_OPTS=...               → respects --force-ipv4/--force-ipv6 if present
        """
        try:
            # Player client selection
            extractor_args = (ydl_opts.get('extractor_args') or {}).copy()
            yt_args = dict((extractor_args.get('youtube') or {}))

            force_client = (os.getenv('YTDLP_FORCE_CLIENT') or '').strip().lower()
            safe_mode = (os.getenv('YTDLP_SAFE_MODE') or '').strip().lower() in {'1', 'true', 'yes'}
            if force_client in {'android', 'web', 'tv', 'web_safari'}:
                yt_args['player-client'] = [force_client]
            elif safe_mode:
                yt_args['player-client'] = ['android']

            if yt_args:
                extractor_args['youtube'] = yt_args
                ydl_opts['extractor_args'] = extractor_args

            # Sleep / pacing
            sleep_req = os.getenv('YTDLP_SLEEP_REQUESTS')
            if sleep_req and sleep_req.isdigit():
                v = max(0, int(sleep_req))
                ydl_opts['sleep_interval'] = v
                ydl_opts['max_sleep_interval'] = max(v, ydl_opts.get('max_sleep_interval') or v)
                ydl_opts['sleep_interval_requests'] = v
                ydl_opts['sleep_interval_subtitles'] = v

            # Retries
            retries = os.getenv('YTDLP_RETRIES')
            if retries and retries.isdigit():
                r = max(0, int(retries))
                ydl_opts['retries'] = r
                ydl_opts['fragment_retries'] = r

            # Cookies
            cookiefile = os.getenv('YTDLP_COOKIES_FILE')
            if cookiefile and os.path.isfile(cookiefile):
                ydl_opts['cookiefile'] = cookiefile

            # IP stack selection via env
            force_stack = (os.getenv('YTDLP_FORCE_STACK') or '').strip().lower()
            if force_stack in {'4', 'ipv4', 'v4'}:
                ydl_opts['source_address'] = '0.0.0.0'
            elif force_stack in {'6', 'ipv6', 'v6'}:
                ydl_opts['source_address'] = '::'

            # Also honor CLI-style flags present in YT_DLP_OPTS
            cli_opts = os.getenv('YT_DLP_OPTS') or ''
            if '--force-ipv4' in cli_opts and 'source_address' not in ydl_opts:
                ydl_opts['source_address'] = '0.0.0.0'
            if '--force-ipv6' in cli_opts and 'source_address' not in ydl_opts:
                ydl_opts['source_address'] = '::'

        except Exception as _:
            # Do not break extraction on env parsing errors
            pass
    
    def _initialize_llm(self, api_key: str):
        """Initialize the LLM based on provider and model"""
        
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key
            )
        elif self.llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model,
                api_key=api_key
            )
        elif self.llm_provider == "openrouter":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/astrocyte74/stuff",
                    "X-Title": "YouTube Summarizer"
                }
            )
        elif self.llm_provider == "ollama":
            # If a local base is configured, route via our helper and skip ChatOllama
            local_env = os.getenv("TTSHUB_API_BASE") or os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST")
            try:
                print(f"🧠 Ollama base: {self.ollama_base_url}")
                if local_env:
                    print("🧠 Using unified local client (hub/direct); skipping ChatOllama init")
            except Exception:
                pass
            if local_env:
                self.llm = None  # calls will go through modules.ollama_client in _robust_llm_call
            else:
                # Fallback: instantiate ChatOllama only when no local env is defined
                self.llm = ChatOllama(
                    model=self.model,
                    base_url=self.ollama_base_url
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def _ollama_hub_call(self, messages: list, *, max_retries: int = 3) -> str:
        """Call local Ollama via our helper (hub or direct) to avoid path mismatches.

        Expects a single HumanMessage; posts using modules.ollama_client.chat and
        returns the response text.
        """
        # Extract prompt text from LangChain-style messages
        prompt_text = None
        try:
            if isinstance(messages, (list, tuple)) and messages:
                first = messages[0]
                prompt_text = getattr(first, 'content', None) or (first if isinstance(first, str) else None)
        except Exception:
            prompt_text = None
        if not prompt_text:
            raise OllamaClientError("invalid messages for ollama hub call")

        loop = asyncio.get_running_loop()
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                def _call():
                    return oc.chat([{"role": "user", "content": prompt_text}], self.model, stream=False)  # type: ignore
                resp = await loop.run_in_executor(None, _call)
                txt = None
                if isinstance(resp, dict):
                    msg = resp.get('message') or {}
                    txt = (msg or {}).get('content') or resp.get('response')
                if isinstance(txt, str) and txt.strip():
                    return txt
                last_err = OllamaClientError("empty response from ollama hub")
            except Exception as e:
                last_err = e
            await asyncio.sleep(1.2 * (attempt + 1))

        raise OllamaClientError(str(last_err) if last_err else "ollama hub call failed")

    _OLLAMA_PRECHECK_CACHE: dict = {"ok_until": 0.0, "fail_until": 0.0}

    def _ollama_precheck(self, timeout: float = 0.8) -> bool:
        """Quick connectivity probe to local Ollama hub/direct before attempting a long LLM call.

        Uses a tiny HTTP GET to the base API. Caches outcome briefly to avoid spamming checks.
        Returns True if reachable, False if not.
        """
        import time as _time
        now = _time.time()
        cache = self._OLLAMA_PRECHECK_CACHE
        if now < cache.get("ok_until", 0.0):
            return True
        if now < cache.get("fail_until", 0.0):
            return False
        base = (os.getenv("TTSHUB_API_BASE") or os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "").rstrip("/")
        if not base:
            cache["fail_until"] = now + 10.0
            return False
        # Try a cheap endpoint
        import requests as _req
        url = f"{base}/health" if "/api" in base else f"{base}/api/health"
        try:
            r = _req.get(url, timeout=timeout)
            if r.status_code >= 200 and r.status_code < 500:
                cache["ok_until"] = now + 15.0
                return True
        except Exception:
            pass
        cache["fail_until"] = now + 5.0
        try:
            import logging as _logging
            _logging.info("Ollama precheck failed for %s", base)
        except Exception:
            pass
        return False

    def _get_reddit_fetcher(self):
        """Lazily construct and memoize the Reddit fetcher."""
        if RedditFetcher is None:
            raise RedditFetcherError("Reddit support is unavailable – install PRAW and configure credentials.")
        if self._reddit_fetcher is None:
            self._reddit_fetcher = RedditFetcher()
        return self._reddit_fetcher

    def _get_web_fetcher(self):
        """Lazily construct and memoize the web page fetcher."""
        if WebPageFetcher is None:
            raise WebFetcherError(
                "Web summarization support is unavailable – install BeautifulSoup4/readability/trafilatura."
            )
        if self._web_fetcher is None:
            allow_dyn = str(os.getenv('WEB_EXTRACT_ALLOW_DYNAMIC', '0')).strip().lower() in ('1','true','yes')
            self._web_fetcher = WebPageFetcher(allow_dynamic=allow_dyn)
        return self._web_fetcher

    async def process_text_content(
        self,
        *,
        content_id: str,
        text: str,
        metadata: Dict[str, Any],
        summary_type: str = "comprehensive",
        proficiency_level: Optional[str] = None,
        source: str = "web",
        source_metadata: Optional[Dict[str, Any]] = None,
        assume_has_audio: bool = False,
    ) -> Dict[str, Any]:
        """Universal text processing pipeline used by non-YouTube sources."""

        transcript = (text or "").strip()
        if len(transcript) < 50:
            return {
                "error": "No usable content available for summarization.",
                "metadata": metadata,
                "processed_at": datetime.now().isoformat(),
                "content_source": source,
                "id": content_id,
            }

        safe_metadata = dict(metadata or {})
        safe_metadata.setdefault("title", "Untitled content")
        safe_metadata.setdefault("uploader", safe_metadata.get("author") or "Unknown author")
        safe_metadata.setdefault("upload_date", datetime.utcnow().strftime("%Y%m%d"))
        safe_metadata.setdefault("duration", safe_metadata.get("duration") or 0)
        safe_metadata.setdefault("url", safe_metadata.get("url") or "")
        safe_metadata.setdefault("language", safe_metadata.get("language") or "en")
        safe_metadata.setdefault("published_at", safe_metadata.get("published_at") or datetime.utcnow().isoformat())

        transcript_language = safe_metadata.get("language", "en")

        summary_data = await self.generate_summary(transcript, safe_metadata, summary_type, proficiency_level)

        summary_language = None
        if isinstance(summary_data, dict):
            summary_language = summary_data.get("language")
            if "summary" not in summary_data and isinstance(summary_data.get("audio"), str):
                summary_data["summary"] = summary_data["audio"]

        if isinstance(summary_data, str) and not summary_language:
            summary_language = transcript_language

        print("Analyzing content...")
        try:
            self._notify_status("🔎 Analyzing content & categorizing…")
        except Exception:
            pass
        if isinstance(summary_data, dict):
            summary_text = summary_data.get("summary", "") or ""
            if not summary_text:
                for key in ("comprehensive", "key_insights", "bullet_points", "audio"):
                    candidate = summary_data.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        summary_text = candidate.strip()
                        break
        else:
            summary_text = str(summary_data)

        if not summary_text.strip():
            print("⚠️ Summary text unavailable for analysis. Falling back to content excerpt.")
            summary_text = transcript[:6000]

        analysis_data = await self.analyze_content(summary_text, safe_metadata)
        if summary_language:
            analysis_data["language"] = summary_language
        else:
            summary_language = analysis_data.get("language", transcript_language)

        media = {
            "has_audio": bool(assume_has_audio),
            "audio_duration_seconds": safe_metadata.get("duration", 0) if assume_has_audio else 0,
            "has_transcript": bool(transcript),
            "transcript_chars": len(transcript),
        }

        result = {
            "id": content_id,
            "content_source": source,
            "title": safe_metadata.get("title", "")[:300],
            "canonical_url": safe_metadata.get("url", ""),
            "thumbnail_url": safe_metadata.get("thumbnail", ""),
            "published_at": safe_metadata.get("published_at"),
            "duration_seconds": safe_metadata.get("duration", 0),
            "word_count": len(transcript.split()) if transcript else 0,
            "media": media,
            "source_metadata": {source: source_metadata or {}},
            "analysis": analysis_data,
            "original_language": transcript_language,
            "summary_language": summary_language,
            "audio_language": summary_language if assume_has_audio else None,
            "url": safe_metadata.get("url", ""),
            "metadata": safe_metadata,
            "transcript": transcript,
            "summary": summary_data,
            "processed_at": datetime.now().isoformat(),
            "processor_info": {
                "llm_provider": self.llm_provider,
                "model": getattr(self.llm, "model_name", getattr(self.llm, "model", self.model)),
            },
        }

        summary_for_image = ""
        if isinstance(result.get("summary"), str):
            summary_for_image = result["summary"]
        if not summary_for_image and isinstance(summary_data, dict):
            for key in ("summary", "comprehensive", "bullet_points", "key_insights", "audio"):
                candidate = summary_data.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    summary_for_image = candidate
                    break
        allow_image = False
        if summary_for_image:
            lowered = summary_for_image.strip().lower()
            disallowed_prefixes = (
                "unable to generate",
                "unable to create",
                "summary generation failed",
            )
            if not any(lowered.startswith(prefix) for prefix in disallowed_prefixes):
                allow_image = True

        if summary_image_service.SUMMARY_IMAGE_ENABLED and allow_image:
            try:
                image_meta = await summary_image_service.maybe_generate_summary_image(result)
                if image_meta:
                    result["summary_image"] = image_meta
                    result["summary_image_url"] = (
                        image_meta.get("public_url")
                        or image_meta.get("relative_path")
                        or image_meta.get("path")
                    )
                    variant_entry = image_meta.get("analysis_variant")
                    if variant_entry:
                        analysis_block = result.get("analysis")
                        if not isinstance(analysis_block, dict):
                            analysis_block = {}
                        selected_url = (
                            image_meta.get("public_url")
                            or image_meta.get("relative_path")
                            or image_meta.get("path")
                        )
                        analysis_block = summary_image_service.apply_analysis_variant(
                            analysis_block,
                            variant_entry,
                            selected_url=selected_url,
                            prompt=image_meta.get("prompt"),
                            model=image_meta.get("model"),
                        )
                        result["analysis"] = analysis_block
            except Exception as exc:
                logging.debug("summary image generation skipped: %s", exc)
            if SUMMARY_IMAGE_AI2_ENABLED:
                try:
                    ai2_meta = await summary_image_service.maybe_generate_summary_image(result, mode="ai2")
                    if ai2_meta:
                        result["summary_image_ai2"] = ai2_meta
                        result["summary_image_ai2_url"] = (
                            ai2_meta.get("public_url")
                            or ai2_meta.get("relative_path")
                            or ai2_meta.get("path")
                        )
                        variant_entry = ai2_meta.get("analysis_variant")
                        if variant_entry:
                            analysis_block = result.get("analysis")
                            analysis_block = summary_image_service.apply_analysis_variant(
                                analysis_block,
                                variant_entry,
                                prompt=ai2_meta.get("prompt"),
                                model=ai2_meta.get("model"),
                            )
                            result["analysis"] = analysis_block
                except Exception as exc:
                    logging.debug("summary ai2 image generation skipped: %s", exc)

        return result

    async def process_reddit_thread(
        self,
        reddit_url: str,
        summary_type: str = "comprehensive",
        proficiency_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a Reddit submission and run it through the summarization pipeline."""
        try:
            fetcher = self._get_reddit_fetcher()
        except RedditFetcherError as exc:
            return {
                "error": str(exc),
                "url": reddit_url,
                "processed_at": datetime.now().isoformat(),
                "content_source": "reddit",
            }

        try:
            fetched = fetcher.fetch(reddit_url)
        except RedditFetcherError as exc:
            return {
                "error": str(exc),
                "url": reddit_url,
                "processed_at": datetime.now().isoformat(),
                "content_source": "reddit",
            }

        data = fetched.to_dict()
        content_text = data["combined_text"]
        content_id = f"reddit:{data['id']}"

        created_dt = datetime.fromtimestamp(data["created_utc"], tz=timezone.utc)
        upload_date = created_dt.strftime("%Y%m%d")

        metadata = {
            "title": data["title"],
            "uploader": f"u/{data['author']}",
            "author": data["author"],
            "channel": f"r/{data['subreddit']}",
            "channel_id": data["subreddit"],
            "upload_date": upload_date,
            "duration": 0,
            "url": data["canonical_url"],
            "language": data.get("language") or "en",
            "published_at": data["created_at_iso"],
            "thumbnail": data.get("thumbnail"),
            "subreddit": data["subreddit"],
            "score": data["score"],
            "upvote_ratio": data["upvote_ratio"],
            "num_comments": data["num_comments"],
            "flair": data.get("flair"),
            "comment_snippets": data.get("comment_snippets", []),
        }

        source_metadata = {
            "submission_id": data["id"],
            "subreddit": data["subreddit"],
            "author": data["author"],
            "score": data["score"],
            "upvote_ratio": data["upvote_ratio"],
            "num_comments": data["num_comments"],
            "flair": data.get("flair"),
            "permalink": data["canonical_url"],
            "comment_samples": data.get("comment_snippets", []),
            "selftext_length": len(data.get("selftext", "")),
        }

        return await self.process_text_content(
            content_id=content_id,
            text=content_text,
            metadata=metadata,
            summary_type=summary_type,
            proficiency_level=proficiency_level,
            source="reddit",
            source_metadata=source_metadata,
            assume_has_audio=False,
        )

    async def process_web_page(
        self,
        web_url: str,
        summary_type: str = "comprehensive",
        proficiency_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a web article and run it through the summarization pipeline."""

        try:
            fetcher = self._get_web_fetcher()
        except WebFetcherError as exc:
            return {
                "error": str(exc),
                "url": web_url,
                "processed_at": datetime.now().isoformat(),
                "content_source": "web",
            }

        try:
            fetched = fetcher.fetch(web_url)
        except WebFetcherError as exc:
            return {
                "error": str(exc),
                "url": web_url,
                "processed_at": datetime.now().isoformat(),
                "content_source": "web",
            }

        content = fetched.content
        text = fetched.text
        content_id = content.id
        normalized_id = content_id.split(":", 1)[-1]

        canonical_url = content.canonical_url or web_url
        parsed_url = urlparse(canonical_url or web_url or "")
        hostname = (parsed_url.hostname or parsed_url.netloc or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if ":" in hostname:
            hostname = hostname.split(":", 1)[0]

        def _humanize_hostname(host: str) -> str:
            if not host:
                return ""
            tokens = host.replace("_", "-").split(".")
            filtered = [t for t in tokens if t and t not in {"www", "com", "org", "net", "co"}]
            if not filtered:
                filtered = [t for t in tokens if t]
            label = " ".join(part.replace("-", " ") for part in filtered).strip()
            return label.title() if label else host

        site_label = (content.site_name or "").strip()
        if not site_label:
            site_label = _humanize_hostname(hostname)

        author = (content.author or "").strip()
        channel_display = site_label or author or "Web article"
        uploader_display = author or site_label or "Web article"

        metadata = {
            "title": content.title or "Untitled article",
            "uploader": uploader_display,
            "author": author,
            "channel": channel_display,
            "channel_id": hostname or channel_display,
            "duration": 0,
            "url": canonical_url or web_url,
            "language": content.language or "en",
            "published_at": content.published_at,
            "thumbnail": content.top_image,
            "video_id": normalized_id,
            "content_id": content_id,
        }

        source_metadata = {
            "id": content.id,
            "canonical_url": content.canonical_url,
            "site_name": content.site_name,
            "title": content.title,
            "language": content.language,
            "author": content.author,
            "published_at": content.published_at,
            "top_image": content.top_image,
            "video_id": normalized_id,
            "site_hostname": hostname,
            "site_label": site_label or channel_display,
            "extractor_notes": content.extractor_notes,
        }

        # Wikipedia fast-path: sectioned Insights/consolidation when enabled
        try:
            wiki_mode = (os.getenv("WIKI_API_MODE") or "auto").strip().lower()
        except Exception:
            wiki_mode = "auto"
        is_wikipedia = (site_label.lower() == "wikipedia" or hostname.endswith("wikipedia.org"))
        normalized_type = (summary_type or "").strip().lower()
        if is_wikipedia and wiki_mode not in {"off", "disabled"} and normalized_type in {"key-insights", "insights"}:
            try:
                result = await self._process_wikipedia_insights(
                    content=content,
                    raw_text=text,
                    metadata=metadata,
                    source_metadata=source_metadata,
                )
                return result
            except Exception as exc:
                print(f"⚠️ Wikipedia fast-path failed, falling back to generic pipeline: {exc}")

        return await self.process_text_content(
            content_id=content_id,
            text=text,
            metadata=metadata,
            summary_type=summary_type,
            proficiency_level=proficiency_level,
            source="web",
            source_metadata=source_metadata,
            assume_has_audio=False,
        )

    # --- Wikipedia sectioned Insights pipeline ------------------------ #
    def _wiki_parse_sections(self, html: str) -> List[Dict[str, str]]:
        """Return a list of {title, text} sections from Wikipedia mobile-html or summary html.

        - Includes the lead section as title 'Overview' if present
        - Skips boilerplate sections like 'References', 'See also', 'Notes'
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html or "", "html.parser")
        sections: List[Dict[str, str]] = []
        SKIP_TITLES = {"references", "see also", "notes", "external links", "further reading", "bibliography"}

        # Lead (data-mw-section-id="0")
        for lead in soup.find_all("section", attrs={"data-mw-section-id": "0"}):
            # Strip references/superscripts in lead
            for sup in lead.find_all("sup"):
                sup.decompose()
            lead_text = lead.get_text("\n", strip=True)
            if lead_text and len(lead_text) > 120:
                sections.append({"title": "Overview", "text": lead_text})

        # Subsequent sections
        for sec in soup.find_all("section"):
            sid = sec.get("data-mw-section-id")
            if not sid or sid == "0":
                continue
            h = sec.find(["h2", "h3"]) or None
            title = (h.get_text(" ", strip=True) if h else "").strip()
            if not title:
                continue
            if title.lower() in SKIP_TITLES:
                continue
            # Clean references
            for sup in sec.find_all("sup"):
                sup.decompose()
            body_text = sec.get_text("\n", strip=True)
            # Remove the heading from body if duplicated at start
            if body_text.startswith(title):
                body_text = body_text[len(title):].lstrip(" \n\t:-")
            if len(body_text) < 60:
                continue
            sections.append({"title": title, "text": body_text})
        return sections

    def _wiki_trim_sections(self, sections: List[Dict[str, str]], limit: int = 8, per_len: int = 2200) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for sec in sections:
            if len(out) >= max(1, limit):
                break
            text = sec.get("text", "")
            if len(text) > per_len:
                # Trim at sentence boundary if possible
                snippet = text[:per_len]
                cut = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
                if cut > per_len * 0.6:
                    text = snippet[: cut + 1]
                else:
                    text = snippet + "…"
            out.append({"title": sec.get("title", ""), "text": text})
        return out

    async def _wiki_section_bullets(self, title: str, body: str, language: str) -> str:
        prompt = f"""
Summarize only the following Wikipedia section.

Article: {title}

Section text:
{self._sanitize_content(body, max_length=4000)}

Output:
- 3–6 “• ” bullets with concrete facts, names, dates, figures.

Rules:
- Use only information explicitly present; no speculation.
- Keep each bullet ≤ 18 words.
- Use the article's language ({language}).
- No headings, no code fences, no emojis.
"""
        # For wiki per‑section calls, keep retries minimal; the local hub has its own backoff.
        return (await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="wiki section bullets", max_retries=1)) or ""

    async def _wiki_consolidate_insights(self, article_title: str, section_bullets: List[Dict[str, str]], language: str) -> str:
        parts = []
        for sb in section_bullets:
            if sb.get("bullets"):
                parts.append(f"## {sb['title']}\n{sb['bullets']}")
        joined = "\n\n".join(parts)
        prompt = f"""
You are consolidating a set of per‑section bullet summaries from a Wikipedia article.

Article: {article_title}

Section summaries:
{joined}

Task: Organize into 3–5 content‑derived categories with concise headings. Under each heading, include 3–5 “• ” bullets capturing concrete facts, names, or metrics. Do not add a final “Bottom line”.

Rules:
- Use only given bullets; merge duplicates; no speculation.
- Keep each bullet ≤ 18 words.
- Use {language}.
- No code fences or emojis.
"""
        # Consolidation is lightweight; avoid noisy retry logs.
        return (await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="wiki insights consolidation", max_retries=1)) or ""

    async def _process_wikipedia_insights(
        self,
        *,
        content,
        raw_text: str,
        metadata: Dict[str, Any],
        source_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sectioned two‑pass Insights specifically for Wikipedia articles."""
        html = getattr(content, "html", None) or ""
        language = (content.language or metadata.get("language") or "en")
        all_sections = self._wiki_parse_sections(html)
        # Limits
        try:
            sec_limit = int(os.getenv("WIKI_SECTION_LIMIT", "8"))
        except Exception:
            sec_limit = 8
        try:
            per_len = int(os.getenv("WIKI_SECTION_MAX_CHARS", "2200"))
        except Exception:
            per_len = 2200
        sections = self._wiki_trim_sections(all_sections, limit=sec_limit, per_len=per_len)
        try:
            logging.info(
                "Wikipedia pipeline: '%s' -> %d sections (limit=%d, per_len=%d); provider=%s model=%s",
                metadata.get("title", "Article"),
                len(sections),
                sec_limit,
                per_len,
                self.llm_provider,
                self.model,
            )
        except Exception:
            pass

        # Fallback: if we failed to split, fall back to generic pipeline
        if not sections:
            return await self.process_text_content(
                content_id=content.id,
                text=raw_text,
                metadata=metadata,
                summary_type="key-insights",
                proficiency_level=None,
                source="web",
                source_metadata=source_metadata,
                assume_has_audio=False,
            )

        # First pass: bullets per section (parallel-ish but sequential here)
        section_bullets: List[Dict[str, str]] = []
        for sec in sections:
            bullets = await self._wiki_section_bullets(sec["title"], sec["text"], language)
            section_bullets.append({"title": sec["title"], "bullets": bullets})

        # Second pass: consolidate into Insights
        insights_text = await self._wiki_consolidate_insights(metadata.get("title", "Wikipedia Article"), section_bullets, language)
        if not insights_text.strip():
            # Fallback: join section bullets as a single list under generic heading
            merged = "\n\n".join(sb.get("bullets", "") for sb in section_bullets if sb.get("bullets"))
            insights_text = merged or raw_text[:2000]

        # Headline
        title_prompt = f"""
Write a single, specific headline (12–16 words, no emojis, no colon) that states subject and concrete value.
IMPORTANT: Respond in {language}.
Source title: {metadata.get('title','')}
Preview:
{insights_text[:1000]}
"""
        headline_text = await self._robust_llm_call([HumanMessage(content=title_prompt)], operation_name="wiki headline", max_retries=1)
        headline_text = headline_text or metadata.get("title", "Generated Summary")

        # Build result in the same shape as process_text_content
        summary_data = {
            "summary": insights_text,
            "headline": headline_text,
            "summary_type": "key-insights",
            "generated_at": datetime.now().isoformat(),
            "language": language,
        }

        # Analyze content (reusing the same analyzer)
        summary_text_for_analysis = insights_text
        analysis_data = await self.analyze_content(summary_text_for_analysis, metadata)

        media = {
            "has_audio": False,
            "audio_duration_seconds": 0,
            "has_transcript": bool(raw_text),
            "transcript_chars": len(raw_text or ""),
        }

        result = {
            "id": content.id,
            "content_source": "web",
            "title": metadata.get("title", ""),
            "canonical_url": metadata.get("url", ""),
            "thumbnail_url": metadata.get("thumbnail", ""),
            "published_at": metadata.get("published_at"),
            "duration_seconds": metadata.get("duration", 0),
            "word_count": len((raw_text or "").split()) if raw_text else 0,
            "media": media,
            "source_metadata": {"web": source_metadata or {}},
            "analysis": analysis_data,
            "original_language": language,
            "summary_language": language,
            "audio_language": None,
            "url": metadata.get("url", ""),
            "metadata": metadata,
            "transcript": raw_text,
            "summary": summary_data,
            "processed_at": datetime.now().isoformat(),
            "processor_info": {
                "llm_provider": self.llm_provider,
                "model": getattr(self.llm, "model_name", getattr(self.llm, "model", self.model)),
            },
        }

        # Optional: image generation remains as in process_text_content
        summary_for_image = insights_text
        allow_image = bool(summary_for_image and not summary_for_image.lower().startswith("unable to "))
        if summary_image_service.SUMMARY_IMAGE_ENABLED and allow_image:
            try:
                image_meta = await summary_image_service.maybe_generate_summary_image(result)
                if image_meta:
                    result["summary_image"] = image_meta
                    result["summary_image_url"] = (
                        image_meta.get("public_url")
                        or image_meta.get("relative_path")
                        or image_meta.get("path")
                    )
                    variant_entry = image_meta.get("analysis_variant")
                    if variant_entry:
                        analysis_block = result.get("analysis")
                        if not isinstance(analysis_block, dict):
                            analysis_block = {}
                        selected_url = (
                            image_meta.get("public_url")
                            or image_meta.get("relative_path")
                            or image_meta.get("path")
                        )
                        analysis_block = summary_image_service.apply_analysis_variant(
                            analysis_block,
                            variant_entry,
                            selected_url=selected_url,
                            prompt=image_meta.get("prompt"),
                            model=image_meta.get("model"),
                        )
                        result["analysis"] = analysis_block
            except Exception as exc:
                logging.debug("summary image generation skipped: %s", exc)

        return result
    
    def _extract_with_robust_ytdlp(self, youtube_url: str, ydl_opts: dict, attempt: int = 1) -> Optional[dict]:
        """Extract info using yt-dlp with robust error handling and retries"""
        import random
        import time
        
        max_attempts = 3
        base_delay = 2
        
        try:
            if attempt > 1:
                jitter = random.uniform(0.5, 1.5)
                print(f"🔄 yt-dlp retry attempt {attempt} in {jitter:.1f}s...")
                time.sleep(jitter)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(youtube_url, download=False)
                
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            normalized_error = error_msg.lower()

            # Fallback: try again with stripped-down options when formats aren't available
            format_error_triggers = (
                "requested format is not available" in normalized_error
                or "not available on this app" in normalized_error
                or "nsig" in normalized_error
            )
            if format_error_triggers and not ydl_opts.get('_format_fallback_applied'):
                logging.warning("🛠️ yt-dlp: format unavailable, retrying with generic options")
                fallback_opts = deepcopy(ydl_opts)
                fallback_opts['_format_fallback_applied'] = True
                # Remove subtitle/metadata writes to simplify request footprint
                fallback_opts.pop('writeautomaticsub', None)
                fallback_opts.pop('writesubtitles', None)
                fallback_opts.pop('writeinfojson', None)
                fallback_opts.pop('subtitleslangs', None)
                fallback_opts.pop('subtitlesformat', None)
                fallback_opts['format'] = 'best'
                fallback_opts['noplaylist'] = True
                fallback_opts['simulate'] = True

                extractor_args = fallback_opts.get('extractor_args') or {}
                if 'youtube' in extractor_args:
                    youtube_args = extractor_args['youtube'].copy()
                    youtube_args.pop('player-client', None)
                    if youtube_args:
                        extractor_args['youtube'] = youtube_args
                    else:
                        extractor_args.pop('youtube', None)
                    fallback_opts['extractor_args'] = extractor_args

                return self._extract_with_robust_ytdlp(youtube_url, fallback_opts, attempt + 1)

            if "403" in error_msg or "Forbidden" in error_msg:
                print(f"🚫 Rate limited (403) on attempt {attempt}")
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    print(f"⏳ Backoff: waiting {delay:.1f}s...")
                    time.sleep(delay)
                    return self._extract_with_robust_ytdlp(youtube_url, ydl_opts, attempt + 1)
                    
            elif "429" in error_msg or "Too Many Requests" in error_msg:
                print(f"🚫 Too many requests (429) on attempt {attempt}")
                if attempt < max_attempts:
                    delay = base_delay * 2 * (2 ** (attempt - 1)) + random.uniform(0, 3)
                    print(f"⏳ Heavy backoff: waiting {delay:.1f}s...")
                    time.sleep(delay)
                    return self._extract_with_robust_ytdlp(youtube_url, ydl_opts, attempt + 1)
                    
            elif "unavailable" in error_msg.lower() or "private" in error_msg.lower():
                print(f"⚠️ Video unavailable or private: {error_msg}")
                return None
                
                logging.warning(f"⚠️ yt-dlp metadata extraction failed: {error_msg}")
                return None
            
        except Exception as e:
            print(f"❌ Unexpected error in yt-dlp: {e}")
            return None
    
    def _get_fallback_metadata(self, youtube_url: str, video_id: str) -> dict:
        """Get metadata by scraping the watch page when yt-dlp fails."""
        try:
            import json
            import re
            import requests
            from bs4 import BeautifulSoup

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(youtube_url, headers=headers, timeout=10)
            response.raise_for_status()

            html_text = response.text
            soup = BeautifulSoup(html_text, 'html.parser')

            title = 'Unknown Title'
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().replace(' - YouTube', '').strip()

            uploader = 'Unknown'
            channel_meta = soup.find('meta', {'name': 'author'})
            if channel_meta:
                uploader = channel_meta.get('content', 'Unknown')

            player_response = None
            match = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', html_text, re.DOTALL)
            if match:
                try:
                    player_response = json.loads(match.group(1))
                except json.JSONDecodeError:
                    player_response = None

            video_details = player_response.get('videoDetails', {}) if player_response else {}

            duration_seconds = int(video_details.get('lengthSeconds', '0')) if video_details.get('lengthSeconds') else 0
            view_count = int(video_details.get('viewCount', '0')) if video_details.get('viewCount') else 0
            channel_id = video_details.get('channelId', '')
            uploader = video_details.get('author', uploader)
            description = video_details.get('shortDescription', '')
            publish_date = video_details.get('publishDate', '')

            return {
                'title': video_details.get('title', title),
                'description': description,
                'uploader': uploader,
                'upload_date': publish_date,
                'duration': duration_seconds,
                'duration_string': str(duration_seconds) if duration_seconds else 'Unknown',
                'view_count': view_count,
                'url': youtube_url,
                'video_id': video_id,
                'id': video_id,
                'channel_url': f'https://www.youtube.com/channel/{channel_id}' if channel_id else '',
                'tags': video_details.get('keywords', []),
                'thumbnail': f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg' if video_id else '',
                'like_count': 0,
                'comment_count': 0,
                'channel_follower_count': 0,
                'uploader_id': channel_id,
                'uploader_url': f'https://www.youtube.com/channel/{channel_id}' if channel_id else '',
                'categories': [],
                'availability': 'public',
                'live_status': 'live' if video_details.get('isLive') else 'not_live',
                'age_limit': 0,
                'resolution': '',
                'fps': 0,
                'aspect_ratio': 0.0,
                'vcodec': '',
                'acodec': '',
                'automatic_captions': [],
                'subtitles': [],
                'release_timestamp': 0,
                'chapters': [],
            }
        except Exception as e:
            logging.warning(f"⚠️ Fallback metadata extraction failed: {str(e)}")

        return {
            'title': f'Video {video_id}',
            'description': '',
            'uploader': 'Unknown',
            'upload_date': '',
            'duration': 0,
            'duration_string': 'Unknown',
            'view_count': 0,
            'url': youtube_url,
            'video_id': video_id,
            'id': video_id,
            'channel_url': '',
            'tags': [],
            'thumbnail': f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg' if video_id else '',
            'chapters': [],
        }

    def _extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'watch?v=' in youtube_url:
            return youtube_url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            return youtube_url.split('youtu.be/')[1].split('?')[0]
        elif '/embed/' in youtube_url:
            return youtube_url.split('/embed/')[1].split('?')[0]
        else:
            # Assume it's already a video ID
            return youtube_url
    
    def _get_transcript_and_metadata_via_api(self, video_id: str, youtube_url: str) -> dict:
        """Extract transcript and attempt basic metadata using youtube-transcript-api + web scraping"""
        transcript_text = None
        transcript_language = None
        transcript_source = 'none'
        
        @contextlib.contextmanager
        def _transcript_proxy_context():
            """Temporarily route only transcript HTTP calls via a proxy.

            If YT_TRANSCRIPT_PROXY is set, this context sets HTTPS_PROXY/HTTP_PROXY
            (and lowercase variants) during the transcript fetch, then restores them.
            """
            proxy = os.getenv("YT_TRANSCRIPT_PROXY")
            if not proxy:
                yield
                return
            try:
                logging.info("Transcript proxy active: %s", proxy)
            except Exception:
                pass
            keys = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
            old = {k: os.environ.get(k) for k in keys}
            try:
                for k in keys:
                    os.environ[k] = proxy
                yield
            finally:
                for k in keys:
                    val = old.get(k)
                    if val is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = val
        
        # Get transcript via youtube-transcript-api with multiple language support
        if TRANSCRIPT_API_AVAILABLE:
            try:
                with _transcript_proxy_context():
                    # Cache check first
                    cached_text, cached_lang = _transcript_cache_get(video_id)
                    if cached_text:
                        transcript_text = cached_text
                        transcript_language = cached_lang
                        transcript_source = 'cache'
                        raise StopIteration  # Skip fetching

                    api = YouTubeTranscriptApi()
                    
                    # Try multiple language codes in order of preference
                    language_codes = [
                        'en',           # English (US)
                        'en-GB',        # English (UK)
                        'en-US',        # English (US explicit)
                        'en-AU',        # English (Australia)
                        'en-CA',        # English (Canada)
                        'es',           # Spanish
                        'es-ES',        # Spanish (Spain)
                        'es-MX',        # Spanish (Mexico)
                        'fr',           # French
                        'fr-FR',        # French (France)
                        'fr-CA'         # French (Canada)
                    ]
                    
                    # --- Global transcript rate limiter / circuit breaker ---
                    transcript_data = None
                    try:
                        min_interval = float(os.getenv('YT_TRANSCRIPT_MIN_INTERVAL', '20') or '20')
                    except Exception:
                        min_interval = 20.0
                    try:
                        threshold = int(os.getenv('YT_TRANSCRIPT_CIRCUIT_THRESHOLD', '3') or '3')
                    except Exception:
                        threshold = 3
                    try:
                        cooldown = float(os.getenv('YT_TRANSCRIPT_COOLDOWN', '600') or '600')
                    except Exception:
                        cooldown = 600.0

                    global _TRANSCRIPT_LIMITER
                    if _TRANSCRIPT_LIMITER is None:
                        _TRANSCRIPT_LIMITER = TranscriptLimiter(min_interval, threshold, cooldown)
                    else:
                        _TRANSCRIPT_LIMITER.update(min_interval, threshold, cooldown)

                    paused = False
                    if not _TRANSCRIPT_LIMITER.before_request():
                        print("⏸️ Transcript fetch paused by circuit breaker (cooldown active)")
                        paused = True
                    else:
                        for lang_code in language_codes:
                            try:
                                transcript_data = api.fetch(video_id, [lang_code])
                                transcript_language = lang_code
                                _TRANSCRIPT_LIMITER.on_success()
                                transcript_source = 'fetched'
                                break
                            except Exception as e_lang:
                                # Detect 429-style throttling
                                emsg = str(e_lang).lower()
                                if '429' in emsg or 'too many requests' in emsg:
                                    _TRANSCRIPT_LIMITER.on_429()
                                continue
                    
                    if transcript_data:
                        text_parts = [snippet.text for snippet in transcript_data]
                        transcript_text = ' '.join(text_parts)
                        _transcript_cache_put(video_id, transcript_text, transcript_language or '')
                        print(f"✅ YouTube Transcript API: Extracted {len(transcript_text)} characters in {transcript_language}")
                    else:
                        if paused:
                            transcript_source = 'paused'
                        print(f"⚠️ YouTube Transcript API: No supported language found")
            except Exception as e:
                # Swallow StopIteration from cache hit
                if not isinstance(e, StopIteration):
                    print(f"⚠️ YouTube Transcript API failed: {e}")
        
        # --- Metadata: prefer official Data API when configured; else fallback to yt-dlp, else scrape ---
        metadata = None
        raw_source_pref = (os.getenv('YT_METADATA_SOURCE') or 'auto').strip()
        source_pref = raw_source_pref.split(",")[0].strip().lower() if raw_source_pref else "auto"
        api_key = os.getenv('YT_API_KEY')
        metadata_source = 'none'
        combo_pref = source_pref in (
            "api+yt_dlp",
            "api_plus_ytdlp",
            "data_api_plus_ytdlp",
            "both",
        )

        # Helper: YouTube Data API v3 (videos.list)
        def _metadata_via_data_api() -> Optional[dict]:
            import requests
            try:
                if not api_key:
                    return None
                parts = 'snippet,contentDetails,statistics'
                url = (
                    'https://www.googleapis.com/youtube/v3/videos?'
                    f'part={parts}&id={video_id}&key={api_key}'
                )
                r = requests.get(url, timeout=12)
                if r.status_code != 200:
                    return None
                data = r.json() or {}
                items = data.get('items') or []
                if not items:
                    return None
                it = items[0]
                sn = it.get('snippet') or {}
                cd = it.get('contentDetails') or {}
                st = it.get('statistics') or {}
                title = (sn.get('title') or '').strip()
                desc = sn.get('description') or ''
                channel = (sn.get('channelTitle') or 'Unknown').strip()
                channel_id = sn.get('channelId') or ''
                published = (sn.get('publishedAt') or '')
                # Convert ISO8601 duration to seconds
                import re
                dur_iso = cd.get('duration') or ''
                m = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$', dur_iso)
                secs = 0
                if m:
                    h = int(m.group(1) or 0); m_ = int(m.group(2) or 0); s = int(m.group(3) or 0)
                    secs = h*3600 + m_*60 + s
                # Views as int if present
                try:
                    views = int(st.get('viewCount', 0))
                except Exception:
                    views = 0
                # Best thumbnail
                thumbs = (sn.get('thumbnails') or {})
                thumb_url = ''
                for k in ('maxres','standard','high','medium','default'):
                    if k in thumbs and 'url' in thumbs[k]:
                        thumb_url = thumbs[k]['url']
                        break
                # YYYYMMDD
                upload_date = ''
                if published:
                    try:
                        from datetime import datetime
                        upload_date = datetime.fromisoformat(published.replace('Z','+00:00')).strftime('%Y%m%d')
                    except Exception:
                        upload_date = ''
                result = {
                    'title': title or f'Video {video_id}',
                    'description': desc,
                    'uploader': channel or 'Unknown',
                    'upload_date': upload_date,
                    'duration': secs,
                    'duration_string': 'Unknown',
                    'view_count': views,
                    'url': youtube_url,
                    'video_id': video_id,
                    'id': video_id,
                    'channel_url': f'https://www.youtube.com/channel/{channel_id}' if channel_id else '',
                    'tags': sn.get('tags', []),
                    'thumbnail': thumb_url or (f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg' if video_id else ''),
                    # Fields not available via Data API (leave empty/defaults)
                    'like_count': 0,
                    'comment_count': 0,
                    'channel_follower_count': 0,
                    'uploader_id': '',
                    'uploader_url': f'https://www.youtube.com/channel/{channel_id}' if channel_id else '',
                    'categories': [],
                    'availability': 'public',
                    'live_status': 'not_live',
                    'age_limit': 0,
                    'resolution': '',
                    'fps': 0,
                    'aspect_ratio': 0.0,
                    'vcodec': '',
                    'acodec': '',
                    'automatic_captions': [],
                    'subtitles': [],
                    'release_timestamp': 0,
                    'chapters': [],
                }
                return result
            except Exception:
                return None

        # Check metadata cache first (applies regardless of chosen source)
        cached_meta = _metadata_cache_get(video_id)
        used_api = False
        if cached_meta is not None:
            metadata = cached_meta
            metadata_source = 'cache'
        
        if metadata is None:
            used_api = False
            if source_pref in ('data_api','api') or (source_pref == 'auto' and api_key):
                metadata = _metadata_via_data_api()
                used_api = metadata is not None
                if used_api:
                    logging.info("YouTube metadata: using Data API (videos.list)")
                    metadata_source = 'data_api'
                    _metadata_cache_put(video_id, metadata)

        # Optional: augment Data API metadata with chapters via yt-dlp (single extra call)
        if used_api and combo_pref:
            try:
                ydl_opts = {
                    'skip_download': True,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'ignoreconfig': True,
                    'noplaylist': True,
                    'simulate': True,
                    'format': 'best',
                    'extractor_args': {'youtube': {'player-client': ['android', 'web']}},
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                    },
                    'geo_bypass': True,
                    'geo_bypass_country': 'US',
                    'retries': 3,
                    'socket_timeout': 20,
                }
                self._apply_ytdlp_env(ydl_opts)
                info = self._extract_with_robust_ytdlp(youtube_url, ydl_opts)
                chapters = info.get('chapters') if info else None
                if chapters:
                    metadata['chapters'] = chapters
                    metadata_source = 'data_api+yt_dlp_chapters'
                    _metadata_cache_put(video_id, metadata)
                    logging.info("YouTube metadata: merged Data API with yt-dlp chapters")
            except Exception as exc:
                logging.warning("yt-dlp chapter augment failed: %s", exc)

        # Fallback to yt-dlp when requested or when Data API not used/failed
        if metadata is None and (source_pref in ('yt_dlp','ytdlp','dlp','auto','') or source_pref not in ('data_api','api')):
            try:
                # Enhanced robustness for metadata-only extraction
                ydl_opts = {
                    'skip_download': True,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'ignoreconfig': True,  # Ignore any system/user config that might force incompatible formats
                    'noplaylist': True,
                    'simulate': True,
                    'format': 'best',
                    # Robustness from OpenAI suggestions
                    'extractor_args': {
                        'youtube': {
                            'player-client': ['android', 'web']  # Allow fallback to multiple official clients
                        }
                    },
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                    },
                    'geo_bypass': True,
                    'geo_bypass_country': 'US',
                    'retries': 3,
                    'socket_timeout': 20,
                }
                # Apply environment-driven adjustments (safe mode, retries, cookies, IP stack)
                self._apply_ytdlp_env(ydl_opts)
                # Use robust extraction with retry logic
                info = self._extract_with_robust_ytdlp(youtube_url, ydl_opts)
                if not info:
                    logging.warning("⚠️ yt-dlp metadata extraction failed after retries; continuing with fallback metadata")
                    info = None

                if info:
                    thumb_id = info.get('id', video_id)
                    metadata = {
                        'title': info.get('title', 'Unknown'),
                        'description': info.get('description', ''),
                        'uploader': info.get('uploader', 'Unknown'), 
                        'upload_date': info.get('upload_date', ''),
                        'duration': info.get('duration', 0),
                        'duration_string': info.get('duration_string', 'Unknown'),
                        'view_count': info.get('view_count', 0),
                        'url': youtube_url,
                        'video_id': video_id,
                        'id': info.get('id', video_id),
                        'channel_url': info.get('channel_url', ''),
                        'tags': info.get('tags', []),
                        'thumbnail': info.get('thumbnail') or (f'https://img.youtube.com/vi/{thumb_id}/mqdefault.jpg' if thumb_id else ''),
                        # Enhanced fields from yt-dlp
                        'like_count': info.get('like_count', 0),
                        'comment_count': info.get('comment_count', 0),
                        'channel_follower_count': info.get('channel_follower_count', 0),
                        'uploader_id': info.get('uploader_id', ''),
                        'uploader_url': info.get('uploader_url', ''),
                        'categories': info.get('categories', []),
                        'availability': info.get('availability', 'public'),
                        'live_status': info.get('live_status', 'not_live'),
                        'age_limit': info.get('age_limit', 0),
                        'resolution': info.get('resolution', ''),
                        'fps': info.get('fps', 0),
                        'aspect_ratio': info.get('aspect_ratio', 0.0),
                        'vcodec': info.get('vcodec', ''),
                        'acodec': info.get('acodec', ''),
                        'automatic_captions': list(info.get('automatic_captions', {}).keys()),
                        'subtitles': list(info.get('subtitles', {}).keys()),
                        'release_timestamp': info.get('release_timestamp', 0),
                        'chapters': info.get('chapters') or [],
                    }
                    logging.info("YouTube metadata: using yt-dlp fallback")
                    metadata_source = 'yt_dlp'
                    _metadata_cache_put(video_id, metadata)
                else:
                    metadata = None
            except Exception as e:
                print(f"⚠️ yt-dlp metadata extraction failed: {e}")
                metadata = None

        if metadata is None:
            logging.warning("⚠️ Falling back to HTML scraping for metadata")
            metadata = self._get_fallback_metadata(youtube_url, video_id)
            metadata_source = 'scrape'
            _metadata_cache_put(video_id, metadata)

        # Ensure segments captured if transcript is available
        if transcript_text and transcript_segments is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_segments = YouTubeTranscriptApi.get_transcript(video_id)
                if transcript_segments:
                    _transcript_segment_cache_put(video_id, transcript_segments)
            except Exception:
                transcript_segments = None
        
        try:
            logging.info(
                "YouTube audit: transcript=%s (%d chars) metadata=%s",
                transcript_source,
                len(transcript_text or ''),
                metadata_source,
            )
        except Exception:
            pass

        return {
            'transcript': transcript_text,
            'transcript_language': transcript_language,
            'metadata': metadata,
            'transcript_segments': transcript_segments,
        }
    
    def _try_yt_dlp_transcript_extraction(self, info: dict) -> Optional[str]:
        """Try to extract transcript using yt-dlp as fallback method"""
        transcript_text = ""
        
        try:
            # Check for automatic captions first - try multiple English variants
            english_keys = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            auto_caps = None
            
            if 'automatic_captions' in info:
                for lang_key in english_keys:
                    if lang_key in info['automatic_captions']:
                        auto_caps = info['automatic_captions'][lang_key]
                        print(f"🔍 yt-dlp fallback: Found {len(auto_caps)} automatic caption formats in {lang_key}")
                        break
            
            if auto_caps:
                for cap in auto_caps:
                    if cap.get('ext') in ['srv3', 'json3', 'ttml', 'vtt']:
                        transcript_url = cap['url']
                        try:
                            import urllib.request
                            with urllib.request.urlopen(transcript_url) as response:
                                transcript_data = response.read().decode('utf-8')
                                if cap.get('ext') == 'srv3':
                                    transcript_text = self._parse_srv3_transcript(transcript_data)
                                else:
                                    transcript_text = self._parse_generic_transcript(transcript_data)
                                
                                if transcript_text.strip():
                                    print(f"✅ yt-dlp fallback: Extracted via {cap.get('ext')} format")
                                    return transcript_text
                        except Exception as e:
                            print(f"⚠️  yt-dlp fallback error: {e}")
            
            # Try manual subtitles if auto captions failed
            if 'subtitles' in info and 'en' in info['subtitles']:
                manual_subs = info['subtitles']['en']
                print(f"🔍 yt-dlp fallback: Trying {len(manual_subs)} manual subtitle formats")
                for sub in manual_subs:
                    if sub.get('ext') in ['srv3', 'json3', 'ttml', 'vtt']:
                        try:
                            import urllib.request
                            with urllib.request.urlopen(sub['url']) as response:
                                transcript_data = response.read().decode('utf-8')
                                if sub.get('ext') == 'srv3':
                                    transcript_text = self._parse_srv3_transcript(transcript_data)
                                else:
                                    transcript_text = self._parse_generic_transcript(transcript_data)
                                
                                if transcript_text.strip():
                                    print(f"✅ yt-dlp fallback: Extracted via manual {sub.get('ext')} format")
                                    return transcript_text
                        except Exception as e:
                            print(f"⚠️  yt-dlp fallback error: {e}")
            
            return transcript_text if transcript_text.strip() else None
            
        except Exception as e:
            print(f"⚠️  yt-dlp transcript extraction failed: {e}")
            return None
    
    def extract_transcript(self, youtube_url: str) -> Dict[str, Union[str, List[Dict]]]:
        """Extract transcript and metadata from YouTube video using hybrid approach
        
        Uses youtube-transcript-api for transcripts (primary) and yt-dlp for metadata.
        Falls back to yt-dlp transcript extraction if primary fails.
        
        Args:
            youtube_url: URL of the YouTube video
            
        Returns:
            Dictionary containing video metadata and transcript
        """
        # Initialize variables in outer scope
        transcript_text = None
        transcript_segments = None
        video_id = self._extract_video_id(youtube_url)
        
        # STEP 1: Try simplified approach - transcript API + web scraping for metadata
        result = self._get_transcript_and_metadata_via_api(video_id, youtube_url)
        transcript_text = result['transcript']
        metadata = result['metadata']
        transcript_language = result.get('transcript_language') or 'en'
        # Optional: capture timestamped segments if available
        transcript_segments = _transcript_segment_cache_get(video_id)
        if transcript_text and transcript_segments is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_segments = YouTubeTranscriptApi.get_transcript(video_id)
                if transcript_segments:
                    _transcript_segment_cache_put(video_id, transcript_segments)
            except Exception:
                transcript_segments = None
        
        # If we got a good transcript, return immediately with web-scraped metadata
        if transcript_text and len(transcript_text.strip()) > 100:
            return {
                'metadata': metadata,
                'transcript': transcript_text,
                'transcript_segments': transcript_segments,
                'content_type': 'transcript',
                'transcript_language': transcript_language,
                'success': True,
                'chapters': metadata.get('chapters', []),
            }
        
        # STEP 2: Fallback to yt-dlp if youtube-transcript-api failed
        try:
            print("🔄 Falling back to yt-dlp for both transcript and metadata...")
            # ENHANCED ROBUST YT-DLP CONFIGURATION (OpenAI Suggestions)
            ydl_opts = {
                # Basic extraction settings
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'ignoreconfig': True,  # Avoid surprises from global yt-dlp config (e.g., forced formats)
                
                # ROBUSTNESS ENHANCEMENTS FROM OPENAI SUGGESTIONS
                
                # 1. Android client preference to avoid desktop detection
                'extractor_args': {
                    'youtube': {
                        'player-client': ['android', 'web'],  # Cycle through resilient clients
                    }
                },
                
                # 2. Modern browser headers for better compatibility
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                
                # 3. Geo-bypass for region restrictions
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                
                # 4. Enhanced retry logic with exponential backoff
                'retries': 5,
                'fragment_retries': 5,
                'retry_sleep_functions': {
                    'http': lambda x: min(300, 2 ** x + __import__('random').uniform(0, 1))
                },
                
                # 5. Request rate limiting (gentle for production)
                'sleep_interval': 1,
                'max_sleep_interval': 3,
                'sleep_interval_requests': 1,
                'sleep_interval_subtitles': 1,
                
                # 6. Network timeouts
                'socket_timeout': 30,
                
                # 7. Additional robustness
                'nocheckcertificate': False,
                'prefer_insecure': False,
            }
            
            # Apply environment-driven adjustments (safe mode, retries, cookies, IP stack)
            self._apply_ytdlp_env(ydl_opts)

            # Use robust extraction with comprehensive retry logic
            info = self._extract_with_robust_ytdlp(youtube_url, ydl_opts)
            if not info:
                raise Exception("Robust yt-dlp extraction failed")
            
            # Get basic metadata
            thumb_id = info.get('id') or video_id
            metadata = {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration', 0),
                    'duration_string': info.get('duration_string', ''),
                    'view_count': info.get('view_count', 0),
                    'url': youtube_url,
                    'video_id': info.get('id', ''),
                    'id': info.get('id', ''),
                    'channel_url': info.get('channel_url', ''),
                    'tags': info.get('tags', []),
                    'thumbnail': info.get('thumbnail') or (f"https://img.youtube.com/vi/{thumb_id}/mqdefault.jpg" if thumb_id else ''),
                    
                    # NEW ENHANCED METADATA FIELDS
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'channel_follower_count': info.get('channel_follower_count', 0),
                    'uploader_id': info.get('uploader_id', ''),
                    'uploader_url': info.get('uploader_url', ''),
                    'categories': info.get('categories', []),
                    'availability': info.get('availability', 'public'),
                    'live_status': info.get('live_status', 'not_live'),
                    'age_limit': info.get('age_limit', 0),
                    'resolution': info.get('resolution', ''),
                    'fps': info.get('fps', 0),
                    'aspect_ratio': info.get('aspect_ratio', 0.0),
                    'vcodec': info.get('vcodec', ''),
                    'acodec': info.get('acodec', ''),
                    'automatic_captions': list(info.get('automatic_captions', {}).keys()),
                    'subtitles': list(info.get('subtitles', {}).keys()),
                    'release_timestamp': info.get('release_timestamp', 0),
                    'chapters': info.get('chapters') or [],
            }
            
            # Log enhanced metadata extraction for visibility
            print("📊 ENHANCED METADATA EXTRACTED:")
            print(f"   👍 Engagement: {metadata['like_count']:,} likes, {metadata['comment_count']:,} comments")
            print(f"   👥 Channel: {metadata['uploader']} ({metadata['channel_follower_count']:,} followers)")
            print(f"   🎥 Technical: {metadata['resolution']} @ {metadata['fps']}fps")
            if metadata['automatic_captions']:
                print(f"   🗣️ Captions: {len(metadata['automatic_captions'])} languages available")
            print(f"   🔗 Channel URL: {metadata['uploader_url']}")
            
            # STEP 3: Handle transcript - use primary result or fallback to yt-dlp
            content_type = "transcript"  # Default assumption
            
            if transcript_text and len(transcript_text.strip()) > 100:
                # SUCCESS: We have a good transcript from youtube-transcript-api
                print(f"✅ Using YouTube Transcript API result: {len(transcript_text)} characters")
            else:
                # FALLBACK: Try yt-dlp transcript extraction
                print("🔄 Falling back to yt-dlp transcript extraction...")
                yt_dlp_transcript = self._try_yt_dlp_transcript_extraction(info)
                
                if yt_dlp_transcript and len(yt_dlp_transcript.strip()) > 100:
                    transcript_text = yt_dlp_transcript
                    print(f"✅ yt-dlp transcript fallback successful: {len(transcript_text)} characters")
                elif yt_dlp_transcript:
                    transcript_text = yt_dlp_transcript
                    print(f"⚠️  yt-dlp transcript fallback partial: {len(transcript_text)} characters")
                else:
                    # REJECT: Description-only videos to prevent hallucination
                    description = info.get('description', '')
                    print(f"❌ No transcript available for this video")
                    print(f"❌ Only description available ({len(description)} chars) - rejecting to prevent AI hallucination")
                    return {
                        'error': 'No transcript available - only description found. Rejecting to prevent hallucination.',
                        'success': False,
                        'content_type': 'description_only'
                    }
                
                transcript_language = transcript_language or info.get('language') or 'en'
                return {
                    'metadata': metadata,
                    'transcript': transcript_text,
                    'content_type': content_type,
                    'transcript_language': transcript_language,
                    'success': True,
                    'chapters': metadata.get('chapters', []),
                }
                
        except Exception as e:
            # If yt-dlp fails but we have a transcript from the API, create minimal metadata
            if transcript_text and len(transcript_text.strip()) > 100:
                # Silent fallback - we have transcript, metadata is optional
                video_id = self._extract_video_id(youtube_url)
                
                # Try to get basic metadata from YouTube page as fallback
                print("🔄 Attempting fallback metadata extraction...")
                fallback_metadata = self._get_fallback_metadata(youtube_url, video_id)
                
                return {
                    'metadata': fallback_metadata,
                    'transcript': transcript_text,
                    'content_type': 'transcript',
                    'transcript_language': transcript_language,
                    'success': True,
                    'chapters': fallback_metadata.get('chapters', []),
                }
            else:
                return {
                    'error': str(e),
                    'success': False
                }
    
    def _parse_srv3_transcript(self, srv3_data: str) -> str:
        """Parse SRV3 format transcript data"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(srv3_data)
            
            transcript_lines = []
            for text_elem in root.findall('.//text'):
                text_content = text_elem.text or ""
                if text_content.strip():
                    # Clean up the text
                    text_content = re.sub(r'&[a-zA-Z]+;', '', text_content)
                    text_content = text_content.strip()
                    if text_content:
                        transcript_lines.append(text_content)
            
            return ' '.join(transcript_lines)
        except Exception as e:
            print(f"Error parsing SRV3 transcript: {e}")
            return ""
    
    def _parse_generic_transcript(self, transcript_data: str) -> str:
        """Parse generic transcript formats (JSON3, VTT, TTML)"""
        try:
            transcript_lines = []
            
            # Try JSON format first
            if transcript_data.strip().startswith('{') or transcript_data.strip().startswith('['):
                import json
                try:
                    data = json.loads(transcript_data)
                    if isinstance(data, dict) and 'events' in data:
                        # YouTube JSON3 format
                        for event in data['events']:
                            if 'segs' in event:
                                for seg in event['segs']:
                                    if 'utf8' in seg:
                                        text = seg['utf8'].strip()
                                        if text and text not in ['♪', '[Music]', '[Applause]']:
                                            transcript_lines.append(text)
                    elif isinstance(data, list):
                        # Other JSON formats
                        for item in data:
                            if isinstance(item, dict) and 'text' in item:
                                text = item['text'].strip()
                                if text:
                                    transcript_lines.append(text)
                except json.JSONDecodeError:
                    pass
            
            # Try VTT format
            elif 'WEBVTT' in transcript_data or '-->' in transcript_data:
                lines = transcript_data.split('\n')
                for line in lines:
                    line = line.strip()
                    # Skip timestamps and empty lines
                    if (line and not line.startswith('WEBVTT') and 
                        '-->' not in line and not line.isdigit() and
                        not line.startswith('NOTE') and not line.startswith('STYLE')):
                        # Clean HTML tags if present
                        import re
                        clean_line = re.sub(r'<[^>]+>', '', line)
                        if clean_line.strip():
                            transcript_lines.append(clean_line.strip())
            
            # Try XML/TTML format
            elif '<' in transcript_data and '>' in transcript_data:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(transcript_data)
                    # Find all text elements regardless of namespace
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            text = elem.text.strip()
                            if text not in ['♪', '[Music]', '[Applause]']:
                                transcript_lines.append(text)
                except ET.ParseError:
                    pass
            
            result = ' '.join(transcript_lines)
            return result if result.strip() else ""
            
        except Exception as e:
            print(f"Error parsing generic transcript: {e}")
            return ""
    
    async def generate_summary(self, transcript: str, metadata: Dict, 
                             summary_type: str = "comprehensive", proficiency_level: str = None) -> Dict[str, str]:
        """Generate AI summary of the video transcript with chunking for long content
        
        Args:
            transcript: The video transcript text
            metadata: Video metadata dictionary
            summary_type: Type of summary ('comprehensive', 'audio', 'bullet-points', 'key-insights', 'executive', 'adaptive')
            
        Returns:
            Dictionary containing different summary formats
        """
        
        # Check if transcript needs chunking (>12000 chars to leave room for prompt overhead)
        if len(transcript) > 12000:
            print(f"📄 Long transcript detected ({len(transcript):,} chars) - using chunked processing for full content")
            return await self._generate_chunked_summary(transcript, metadata, summary_type, proficiency_level)
        
        # For shorter transcripts, use direct processing
        print(f"📄 Processing transcript directly ({len(transcript):,} chars)")
        try:
            self._notify_status("📝 Drafting audio summary text…")
        except Exception:
            pass
        
        # Normalize summary_type to canonical keys for robust matching
        TYPE_MAP = {
            "comprehensive": "comprehensive",
            "key points": "key_points", 
            "key_points": "key_points",
            "bullet points": "key_points",
            "bullet-points": "key_points", 
            "insights": "insights",
            "key-insights": "insights",
            "key_insights": "insights",
            "audio summary": "audio",
            "audio-summary": "audio", 
            "audio": "audio",
            "audio-fr": "audio_fr",
            "audio-es": "audio_es",
        }
        normalized_type = TYPE_MAP.get(summary_type.lower().strip(), summary_type)
        
        # Handle multilingual audio with vocabulary logic
        if normalized_type in ("audio_fr", "audio_es"):
            # First create base audio summary in original language
            base_audio_text = await self._create_audio_summary_from_transcript(transcript, metadata)
            if not base_audio_text:
                return {
                    'error': "Failed to generate base audio summary",
                    'summary': "Audio summary generation failed",
                    'headline': "Error generating audio"
                }
            
            # Then translate/adapt with vocabulary support
            target_lang = "French" if normalized_type == "audio_fr" else "Spanish"
            try:
                self._notify_status(f"🌐 Translating to {target_lang}…")
            except Exception:
                pass
            final_audio_text = await self._create_translated_audio_summary(
                base_audio_text, target_lang, proficiency_level
            )
            
            return {
                'audio': final_audio_text,
                'summary': final_audio_text,
                'summary_type': summary_type,
                'generated_at': datetime.now().isoformat(),
                'language': 'fr' if normalized_type == "audio_fr" else 'es'
            }
        
        # Prepare context and prompts based on summary type
        base_context = f"""
        Video Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('uploader', 'Unknown')}
        Upload Date (YYYYMMDD): {metadata.get('upload_date', 'Unknown')}
        Duration (seconds): {metadata.get('duration', 0)}
        URL: {metadata.get('url', 'Unknown')}

        Full Transcript:
        {transcript}
        """
        
        prompts = {
            "comprehensive": f"""
            {base_context}

            Write a comprehensive summary optimized for quick reading.

            Output order:
            1) Overview — 2–3 sentences covering the problem/topic, what’s shown, and the key conclusion.
            2) Sections — 2–4 short, descriptive headings (2–4 words) followed by 3–5 bullets each:
               • Use “• ” bullets. Keep bullets ≤ 18 words.
               • Prefer specifics (data, numbers, names) over generalities.
               • Avoid meta (don’t say “the host says/this video covers”).
            3) Bottom line — one sentence starting “Bottom line: …”.

            Section titles must summarize the key phase or theme (not full sentences). Each bullet must be factual and non‑redundant.

            Rules:
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - Respond in the transcript’s language.
            - Include [mm:ss] timestamps only if explicitly present in transcript text; otherwise omit.
            - If a needed fact isn’t available, write “Unknown”.
            - No code fences, no emojis, no markdown headings.

            Length guidance (not strict):
            - <10 min: 150–220 words; 2 sections.
            - 10–30 min: 220–350 words; 3 sections.
            - >30 min: 350–500 words; 3–4 sections.
            """,

            "audio": f"""
            {base_context}

            Write a natural, text‑to‑speech friendly narration.

            Structure:
            - Opening: 2–3 sentences that jump straight to the substance and high‑level conclusion.
            - Main: Smoothly connect major topics with conversational transitions (no headings/bullets). Use transitions like “First…”, “Next…”, “However…”, “The key trade‑off is…”. Use smooth spoken transitions (“First…”, “Next…”, “However…”, “Finally…”) and vary sentence length for natural rhythm.
            - Closing: One sentence starting “Bottom line: …”.

            Rules:
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - Respond in the transcript’s language.
            - Keep numbers and names accurate; include specific values where present.
            - Avoid list‑like phrasing or enumeration; use implicit transitions instead.
            - No headings, no bullets, no code fences, no emojis.
            - Length: ~180–380 words for most videos; shorter for very short clips.
            """,

            "bullet-points": f"""
            {base_context}

            Produce a skim‑ready bullet summary.

            Output:
            - One‑sentence overview.
            - 10–16 “• ” bullets capturing concrete facts, results, and named items.
            - End with “Bottom line: …”.

            Rules:
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - Respond in the transcript’s language.
            - Each bullet ≤ 18 words; lead with the fact/action.
            - Prefer named entities, figures, and actions over general statements (metrics, model names, versions, dates).
            - Avoid duplication; merge near‑identical points.
            - If the transcript contains comparisons, include at least one bullet explicitly stating the contrast.
            - Timestamps only if explicitly present; else omit.
            - No code fences/emojis/headings.
            """,

            "key-insights": f"""
            {base_context}

            Organize the content into thematic categories based on the material itself (no external analysis).

            Output:
            - 3–5 short category headings derived from the content.
            - Under each heading, provide 3–5 “• ” bullets capturing concrete facts, results, names, or metrics.

            Rules:
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - Respond in the transcript’s language.
            - Keep each bullet ≤ 18 words; lead with the fact/action; avoid duplication.
            - No speculation beyond the transcript; use “Unknown” only when details are missing.
            - Use consistent tone and granularity across categories — each heading should capture a distinct conceptual dimension (e.g., Strategy / Technology / Outcome).
            - No code fences or emojis.
            - Do not add a final “Bottom line”.
            """,
            
            "executive": f"""
            {base_context}

            Format this as a structured EXECUTIVE REPORT. Analyze the content and divide it into 2-4 logical parts based on the video's natural flow.

            **📊 EXECUTIVE SUMMARY**
            2-3 sentences that capture the video's main purpose, key findings, and business/practical value. Professional tone.

            **📋 PART 1: (Give this section a descriptive title)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details  
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **📋 PART 2: (Give this section a descriptive title)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **📋 PART 3: (Give this section a descriptive title - if applicable)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **🎯 STRATEGIC RECOMMENDATIONS**
            • Actionable next step or key takeaway
            • Actionable next step or key takeaway
            • Actionable next step or key takeaway

            Write for senior readers scanning quickly. Front‑load major outcomes and implications before elaborating.

            **Guidelines:**
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - Divide content into 2-4 logical parts (not artificial divisions)
            - Use professional, analytical language
            - Include timestamps where helpful
            - Focus on insights and implications, not just facts
            - Keep each section balanced and substantive
            - Ensure each PART ends with a concise synthesis sentence (1–2 clauses).
            - If content doesn't fit this structure well, adapt the format accordingly
            - British/Canadian spelling; no speculation
            """,
            
            "adaptive": f"""
            {base_context}

            Silently choose the best format for this transcript from:
            - Comprehensive
            - Key Points
            - Key Insights
            - (If clearly procedural) a step‑wise breakdown (6–12 steps, ≤ 12 words/step)

            Output only the chosen format. No meta‑explanation.

            Global rules:
            - Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
            - Prefer short paraphrases of full ideas rather than skipping them entirely.
            - When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
            - Keep events in the original chronological order unless a thematic grouping is requested.
            - Rewrite for clarity and natural flow after compressing, without adding new meaning.
            - If multiple patterns could apply, select the one maximizing clarity for non‑expert readers.
            - Respond in the transcript’s language.
            - Timestamps only if explicitly present; else omit.
            - “Unknown” when information is missing.
            - No code fences/emojis.
            - For Key Points/Insights, use “• ” bullets. For step‑wise, use numbered steps.
            - When choosing step‑wise form, begin with a one‑sentence context statement before numbering.
            - Length guidance: short 120–180 words; dense 250–500 words; step‑wise concise.
            """,
        }
        
        # Generate the summary with robust error handling
        summary_text = await self._robust_llm_call(
            [HumanMessage(content=prompts.get(summary_type, prompts["comprehensive"]))],
            f"Summary generation ({summary_type})"
        )
        
        if not summary_text:
            return {
                'error': "Failed to generate summary after all retries",
                'summary': "Summary generation failed",
                'headline': "Error generating headline"
            }
        
        # Also generate a quick title/headline (using summary for token efficiency)
        title_prompt = f"""
        Write a single, specific headline (12–16 words, no emojis, no colon) that states subject and concrete value.
        Start with a concrete noun or named entity; avoid vague verbs (e.g., "Exploring", "Discussing").
        **IMPORTANT: Respond in the same language as the content.**
        Source title: {metadata.get('title', '')}
        Summary:
        {summary_text[:1200]}
        """
        
        headline_text = await self._robust_llm_call(
            [HumanMessage(content=title_prompt)],
            "Headline generation"
        )
        
        # Enhanced logging to show generated headline
        final_headline = headline_text or "Generated Summary"
        print(f"📝 Generated Headline: {final_headline}")
        
        return {
            'summary': summary_text,
            'headline': final_headline,
            'summary_type': summary_type,
            'generated_at': datetime.now().isoformat(),
            'language': metadata.get('language', 'en')
        }
    
    def _sanitize_content(self, content: str, max_length: int = 8000) -> str:
        """Sanitize and truncate content for safe LLM processing"""
        if not content or not isinstance(content, str):
            return ""
        
        # Remove potentially malicious content
        content = content.replace("```", "").replace("</prompt>", "").replace("<prompt>", "")
        
        # Intelligent truncation - preserve sentences when possible
        if len(content) <= max_length:
            return content
        
        truncated = content[:max_length]
        # Find last sentence boundary within limit
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        sentence_end = max(last_period, last_question, last_exclamation)
        if sentence_end > max_length * 0.8:  # If sentence boundary is reasonably close
            return truncated[:sentence_end + 1]
        
        return truncated + "..."
    
    def _validate_analysis_result(self, analysis: dict) -> dict:
        """Validate and sanitize analysis results according to universal schema"""
        
        # Define allowed values for controlled fields
        allowed_categories = [
            "Education", "Entertainment", "Technology", "AI Software Development", "Computer Hardware", "Home Theater",
            "Business", "Health & Wellness", "How-To & DIY",
            "News & Politics", "Gaming", "Lifestyle", "Science & Nature", "Astronomy", "History",
            "World War I (WWI)", "World War II (WWII)", "Cold War",
            "Arts & Creativity", "Religion & Philosophy", "Sports", "Hobbies & Special Interests",
            "Vlogs & Personal", "Reviews & Products", "General"
        ]
        allowed_complexity = ["Beginner", "Intermediate", "Advanced"]
        allowed_content_types = ["Tutorial", "Review", "Discussion", "News", "Documentary", 
                                "Interview", "Presentation", "Guide"]
        allowed_subcategories = {
            "Education": [
                "Academic Subjects", "Online Learning", "Tutorials & Courses", 
                "Teaching Methods", "Educational Technology", "Student Life"
            ],
            "Entertainment": [
                "Comedy & Humor", "Music & Performance", "Movies & TV", 
                "Celebrity & Pop Culture", "Viral Content", "Reaction Content"
            ],
            "Technology": [
                "Programming & Software Development", "Web Development", "Mobile Development",
                "DevOps & Infrastructure", "Databases & Data Science", "Cybersecurity",
                "Tech Reviews & Comparisons", "Software Tutorials", "Tech News & Trends"
            ],
            "AI Software Development": [
                "Model Selection & Evaluation", "Prompt Engineering & RAG",
                "Training & Fine-Tuning", "Deployment & Serving",
                "Agents & MCP/Orchestration", "APIs & SDKs",
                "Data Engineering & ETL", "Testing & Observability",
                "Security & Safety", "Cost Optimisation"
            ],
            "Computer Hardware": [
                "CPUs & GPUs", "Motherboards & Chipsets",
                "Memory & Storage (SSD/NVMe)", "Cooling & Thermals",
                "Power Supplies", "PC Cases & Builds",
                "Networking & NAS", "Monitors & Peripherals",
                "Benchmarks & Tuning", "Troubleshooting & Repairs"
            ],
            "Home Theater": [
                "AV Receivers & Amplifiers", "Speakers & Subwoofers",
                "Room Correction & Calibration (Dirac, Audyssey)", "Acoustic Treatment",
                "Projectors & Screens", "Media Players & Sources",
                "Cables & Connectivity", "Setups & Tours", "Troubleshooting & Tips"
            ],
            "Business": [
                "Entrepreneurship", "Marketing & Sales", "Finance & Investing", 
                "Career Development", "Leadership & Management", "Industry Analysis"
            ],
            "Health & Wellness": [
                "Fitness & Exercise", "Nutrition & Diet", "Mental Health", 
                "Medical Information", "Lifestyle & Habits", "Alternative Medicine"
            ],
            "How-To & DIY": [
                "Home Improvement", "Crafts & Making", "Repair & Maintenance", 
                "Life Skills", "Creative Projects", "Tools & Techniques"
            ],
            "News & Politics": [
                "Breaking News", "Political Analysis", "Current Events", 
                "Journalism & Reporting", "Government & Policy", "International Affairs"
            ],
            "Gaming": [
                "Game Reviews", "Gameplay & Walkthroughs", "Esports & Competitive", 
                "Game Development", "Gaming Culture", "Retro Gaming"
            ],
            "Lifestyle": [
                "Fashion & Style", "Travel & Adventure", "Food & Cooking", 
                "Relationships", "Personal Development", "Home & Living"
            ],
            "Science & Nature": [
                "Research & Discoveries", "Biology & Life Sciences", "Physics & Chemistry", 
                "Environmental Science", "Nature & Wildlife", "Scientific Method"
            ],
            "Astronomy": [
                "Solar System & Planets", "Stars & Stellar Evolution",
                "Galaxies & Cosmology", "Telescopes & Observing",
                "Space Missions & Exploration", "Astrophotography",
                "Amateur Astronomy", "Space News & Discoveries"
            ],
            "History": [
                "Ancient Civilizations", "Medieval History", "Modern History", 
                "Cultural Heritage", "Historical Analysis", "Biographies"
            ],
            "World War I (WWI)": [
                "Causes & Prelude", "Major Battles & Campaigns", "Home Front & Society",
                "Technology & Weapons", "Diplomacy & Treaties (Versailles)",
                "Biographies & Leaders", "Aftermath & Interwar"
            ],
            "World War II (WWII)": [
                "Causes & Prelude", "European Theatre", "Pacific Theatre",
                "Home Front & Society", "Technology & Weapons", "Intelligence & Codebreaking",
                "Holocaust & War Crimes", "Diplomacy & Conferences (Yalta, Potsdam)",
                "Biographies & Commanders", "Aftermath & Reconstruction"
            ],
            "Cold War": [
                "Origins & Ideologies", "Proxy Wars (Korea, Vietnam, Afghanistan)",
                "Nuclear Strategy & Arms Race", "Espionage & Intelligence",
                "Space Race & Technology", "Domestic Life & Culture",
                "Diplomacy & Crises (Berlin, Cuba)", "Détente & End of Cold War",
                "Leaders & Biographies"
            ],
            "Arts & Creativity": [
                "Visual Arts", "Music Production", "Photography & Video", 
                "Writing & Literature", "Dance & Performance", "Digital Art"
            ],
            "Religion & Philosophy": [
                "Religious Teachings", "Spirituality & Practice", "Philosophical Thought", 
                "Ethics & Morality", "Theological Studies", "Comparative Religion"
            ],
            "Sports": [
                "Game Analysis", "Training & Fitness", "Player Profiles", 
                "Sports News", "Equipment & Gear", "Fantasy Sports"
            ],
            "Hobbies & Special Interests": [
                "Collecting", "Automotive", "Pets & Animals", 
                "Outdoor Activities", "Niche Communities", "Specialized Skills"
            ],
            "Vlogs & Personal": [
                "Daily Life", "Personal Stories", "Opinions & Commentary", 
                "Q&A Sessions", "Behind-the-Scenes", "Life Updates"
            ],
            "Reviews & Products": [
                "Product Reviews", "Unboxings", "Comparisons & Tests", 
                "Service Reviews", "Buying Guides", "Tech Specs"
            ],
            "General": [
                "Miscellaneous", "Mixed Content", "Uncategorized", 
                "General Discussion", "Variety Content", "Other"
            ]
        }
        
        # Validate and clean the analysis
        validated = {
            "category": [],
            "subcategory": None,
            "content_type": "Discussion",
            "complexity_level": "Intermediate", 
            "language": "en",  # Default to English, will be detected separately
            "key_topics": [],
            "named_entities": []
        }
        
        # Validate categories with new subcategories structure
        if "categories" in analysis and isinstance(analysis["categories"], list):
            seen_categories = set()
            validated_category_objects = []
            
            for cat_obj in analysis["categories"][:3]:  # Max 3 categories
                if isinstance(cat_obj, dict) and cat_obj.get("category") in allowed_categories:
                    category_name = cat_obj["category"]
                    if category_name not in seen_categories:
                        # Extract and validate subcategories
                        subcats = cat_obj.get("subcategories", [])
                        if isinstance(subcats, list) and subcats:
                            valid_subcats = [str(sub).strip() for sub in subcats[:3] if str(sub).strip()]
                        else:
                            valid_subcats = ["Other"]
                        
                        validated_category_objects.append({
                            "category": category_name,
                            "subcategories": valid_subcats
                        })
                        seen_categories.add(category_name)
            
            validated["categories"] = validated_category_objects  # New structure
            validated["schema_version"] = 2  # Mark as using structured data
            # Mirror flat names for compatibility
            validated["category"] = [obj["category"] for obj in validated["categories"]]
        
        # Only populate fallback for pre-v2 data
        if validated.get("schema_version", 1) < 2:
            if not validated.get("category"):
                validated["category"] = ["General"]
        elif not validated.get("categories"):
            # Fallback for v2 data without categories
            validated["categories"] = [{"category": "General", "subcategories": ["Other"]}]
            validated["category"] = ["General"]
        
        # Validate content type
        if analysis.get("content_type") in allowed_content_types:
            validated["content_type"] = analysis["content_type"]
        
        # Validate complexity
        if analysis.get("complexity_level") in allowed_complexity:
            validated["complexity_level"] = analysis["complexity_level"]

        # Validate required subcategory (now required for ALL categories)
        if isinstance(analysis.get("subcategory"), str) and validated["category"]:
            primary_cat = validated["category"][0]
            if primary_cat in allowed_subcategories:
                if analysis["subcategory"] in allowed_subcategories[primary_cat]:
                    validated["subcategory"] = analysis["subcategory"]
        
        # Process key topics - convert to slugs
        if "key_topics" in analysis and isinstance(analysis["key_topics"], list):
            topics = []
            for topic in analysis["key_topics"][:7]:  # Max 7 topics
                if isinstance(topic, str) and topic.strip():
                    # Convert to slug format
                    slug = topic.lower().replace(" ", "-").replace("_", "-")
                    # Remove special characters and limit length
                    import re
                    slug = re.sub(r'[^a-z0-9-]', '', slug)[:30]
                    if slug:
                        topics.append(slug)
            validated["key_topics"] = topics[:5]  # Limit to 5 for simplicity
        
        return validated
    
    def _parse_safe_json(self, json_string: str, max_size: int = 10000) -> dict:
        """Safely parse JSON with size limits and validation"""
        if not json_string or len(json_string) > max_size:
            raise ValueError("JSON string too large or empty")

        import json
        import re

        raw = json_string.strip()
        if not raw:
            raise ValueError("JSON string too large or empty")

        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            return result
        except json.JSONDecodeError as original_error:
            # Attempt to sanitize common wrapper patterns (e.g., Markdown code fences)
            candidates = []

            fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
            fence_match = fence_pattern.match(raw)
            if fence_match:
                candidates.append(fence_match.group(1).strip())

            # Handle responses prefixed with a bare "json" identifier
            if raw.lower().startswith('json'):
                candidates.append(raw[4:].strip())

            # Extract substring between the outermost braces as a fallback
            first_brace = raw.find('{')
            last_brace = raw.rfind('}')
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                candidates.append(raw[first_brace:last_brace + 1].strip())

            seen = set()
            for candidate in candidates:
                candidate = candidate.strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue

            raise ValueError(f"Invalid JSON: {str(original_error)}")
    
    async def analyze_content(self, summary: str, metadata: Dict) -> Dict[str, Union[str, List[str]]]:
        """Perform secure content analysis with universal schema support
        
        Enhanced with:
        - Uses summary instead of transcript for token efficiency
        - Input sanitization and validation
        - Safe JSON parsing
        - Universal schema compliance
        - Intelligent content truncation
        """
        
        # Input validation and sanitization
        if not summary or not isinstance(summary, str):
            summary = ""
        if not metadata or not isinstance(metadata, dict):
            metadata = {}
        
        title = str(metadata.get('title', 'Unknown'))[:200]  # Limit title length
        safe_summary = self._sanitize_content(summary, max_length=6000)
        
        # Enhanced analysis prompt with uniform subcategory structure
        analysis_prompt = f"""Analyze this content and extract structured metadata using the hierarchical category system.

Title: {title}
Summary: {safe_summary}

CRITICAL: Every category MUST have a subcategory. Choose the most specific match.

CATEGORIZATION RULES:
1. ALWAYS prefer specialized categories (World War II, AI Software Development) over generic ones (History, Technology)
2. If content mentions WWII, Nazi Germany, SS, Wehrmacht, Hitler, Pearl Harbor, D-Day, Pacific War, Holocaust → use "World War II (WWII)" category
3. If content is about AI/ML programming, models, training, deployment → use "AI Software Development" category
4. Only use "History → Modern History" if it's modern history but NOT specifically WWII
5. Only use "Technology" for general tech content, not AI/ML specific content
6. ASSIGN MULTIPLE CATEGORIES when content genuinely spans different areas (e.g., historical tech video could be both "History" and "Technology")
7. AVOID duplicate categories - each category should be unique
8. Content should have 2-3 categories when it covers multiple distinct topics, but only 1 category if it's clearly focused on one area

CATEGORY STRUCTURE (Category → Required Subcategories):

**Education** → Academic Subjects | Online Learning | Tutorials & Courses | Teaching Methods | Educational Technology | Student Life

**Entertainment** → Comedy & Humor | Music & Performance | Movies & TV | Celebrity & Pop Culture | Viral Content | Reaction Content

**Technology** → Programming & Software Development | Web Development | Mobile Development | DevOps & Infrastructure | Databases & Data Science | Cybersecurity | Tech Reviews & Comparisons | Software Tutorials | Tech News & Trends

**AI Software Development** → Model Selection & Evaluation | Prompt Engineering & RAG | Training & Fine-Tuning | Deployment & Serving | Agents & MCP/Orchestration | APIs & SDKs | Data Engineering & ETL | Testing & Observability | Security & Safety | Cost Optimisation

**Computer Hardware** → CPUs & GPUs | Motherboards & Chipsets | Memory & Storage (SSD/NVMe) | Cooling & Thermals | Power Supplies | PC Cases & Builds | Networking & NAS | Monitors & Peripherals | Benchmarks & Tuning | Troubleshooting & Repairs

**Home Theater** → AV Receivers & Amplifiers | Speakers & Subwoofers | Room Correction & Calibration (Dirac, Audyssey) | Acoustic Treatment | Projectors & Screens | Media Players & Sources | Cables & Connectivity | Setups & Tours | Troubleshooting & Tips

**Business** → Entrepreneurship | Marketing & Sales | Finance & Investing | Career Development | Leadership & Management | Industry Analysis

**Health & Wellness** → Fitness & Exercise | Nutrition & Diet | Mental Health | Medical Information | Lifestyle & Habits | Alternative Medicine

**How-To & DIY** → Home Improvement | Crafts & Making | Repair & Maintenance | Life Skills | Creative Projects | Tools & Techniques

**News & Politics** → Breaking News | Political Analysis | Current Events | Journalism & Reporting | Government & Policy | International Affairs

**Gaming** → Game Reviews | Gameplay & Walkthroughs | Esports & Competitive | Game Development | Gaming Culture | Retro Gaming

**Lifestyle** → Fashion & Style | Travel & Adventure | Food & Cooking | Relationships | Personal Development | Home & Living

**Science & Nature** → Research & Discoveries | Biology & Life Sciences | Physics & Chemistry | Environmental Science | Nature & Wildlife | Scientific Method

**Astronomy** → Solar System & Planets | Stars & Stellar Evolution | Galaxies & Cosmology | Telescopes & Observing | Space Missions & Exploration | Astrophotography | Amateur Astronomy | Space News & Discoveries

**History** → Ancient Civilizations | Medieval History | Modern History | Cultural Heritage | Historical Analysis | Biographies

**World War I (WWI)** → Causes & Prelude | Major Battles & Campaigns | Home Front & Society | Technology & Weapons | Diplomacy & Treaties (Versailles) | Biographies & Leaders | Aftermath & Interwar

**World War II (WWII)** → Causes & Prelude | European Theatre | Pacific Theatre | Home Front & Society | Technology & Weapons | Intelligence & Codebreaking | Holocaust & War Crimes | Diplomacy & Conferences (Yalta, Potsdam) | Biographies & Commanders | Aftermath & Reconstruction

**Cold War** → Origins & Ideologies | Proxy Wars (Korea, Vietnam, Afghanistan) | Nuclear Strategy & Arms Race | Espionage & Intelligence | Space Race & Technology | Domestic Life & Culture | Diplomacy & Crises (Berlin, Cuba) | Détente & End of Cold War | Leaders & Biographies

**Arts & Creativity** → Visual Arts | Music Production | Photography & Video | Writing & Literature | Dance & Performance | Digital Art

**Religion & Philosophy** → Religious Teachings | Spirituality & Practice | Philosophical Thought | Ethics & Morality | Theological Studies | Comparative Religion

**Sports** → Game Analysis | Training & Fitness | Player Profiles | Sports News | Equipment & Gear | Fantasy Sports

**Hobbies & Special Interests** → Collecting | Automotive | Pets & Animals | Outdoor Activities | Niche Communities | Specialized Skills

**Vlogs & Personal** → Daily Life | Personal Stories | Opinions & Commentary | Q&A Sessions | Behind-the-Scenes | Life Updates

**Reviews & Products** → Product Reviews | Unboxings | Comparisons & Tests | Service Reviews | Buying Guides | Tech Specs

**General** → Miscellaneous | Mixed Content | Uncategorized | General Discussion | Variety Content | Other

Return ONLY valid JSON with this exact shape:
{{
  "categories": [
    {{ "category": "<Category>", "subcategories": ["<Subcat 1>", "<Subcat 2>"] }}
  ],
  "languages": {{ "video_language": "en|fr|...", "summary_language": "en|fr|..." }},
  "content_type": "Tutorial|Review|Discussion|News|Documentary|Interview|Presentation|Guide",
  "complexity_level": "Beginner|Intermediate|Advanced",
  "key_topics": ["kebab-case-1", "kebab-case-2", "kebab-case-3"],
  "named_entities": ["Name or Org", "Name or Org"]
}}

Rules:
- Choose the 1-3 BEST categories and subcategories that pertain to this summary. When content covers multiple distinct areas, PREFER using multiple categories over forcing it into just one.
- For EACH chosen category, select 1–3 subcategories FROM ITS OWN allowed list (no cross-category subcats).
- Use exact names from the allowed lists. No nulls/empties/duplicates.
- languages detects both original video language and summary language.
- key_topics are lowercase, hyphenated phrases.

EXAMPLES:
Single Category:
- "SS Weapon" → "category": ["World War II (WWII)"], "subcategory": "Technology & Weapons"
- "Claude Code AI" → "category": ["AI Software Development"], "subcategory": "APIs & SDKs"

Multiple Categories (when content genuinely spans areas):
- "How AI Changed WWII Codebreaking" → "category": ["World War II (WWII)", "AI Software Development"], "subcategory": "Intelligence & Codebreaking"
- "Teaching Python Programming" → "category": ["Education", "Technology"], "subcategory": "Teaching Methods"
- "Tesla Business Analysis" → "category": ["Business", "Technology"], "subcategory": "Industry Analysis\""""
        
        try:
            # Record processing metadata
            processing_start = datetime.now()
            
            analysis_content = await self._robust_llm_call(
                [HumanMessage(content=analysis_prompt)],
                "Content analysis"
            )
            
            if not analysis_content:
                return self._get_fallback_analysis("LLM call failed")
            
            # Safe JSON parsing
            try:
                # Log the raw JSON response from AI
                print(f"🤖 Raw AI JSON Response:")
                print(f"   {analysis_content}")
                
                raw_analysis = self._parse_safe_json(analysis_content)
                validated_analysis = self._validate_analysis_result(raw_analysis)
                
                # Enhanced logging to show AI decisions (from structured data)
                if validated_analysis.get('schema_version') == 2 and 'categories' in validated_analysis:
                    # Show structured categories
                    cats_for_log = [obj["category"] for obj in validated_analysis["categories"]]
                    subs_for_log = {obj["category"]: obj.get("subcategories", []) for obj in validated_analysis["categories"]}
                else:
                    # Fallback to legacy format
                    cats_for_log = validated_analysis.get('category', ['Unknown'])
                    subs_for_log = {'legacy': validated_analysis.get('subcategory', 'None')}
                
                content_type = validated_analysis.get('content_type', 'Unknown')
                complexity = validated_analysis.get('complexity_level', 'Unknown')
                print(f"🎯 AI Analysis Results:")
                print(f"   Categories: {cats_for_log}")
                print(f"   Per-category subcats: {subs_for_log}")
                print(f"   Content Type: {content_type}")  
                print(f"   Complexity: {complexity}")
                print(f"   Schema Version: {validated_analysis.get('schema_version', 1)}")
                
                # Add processing metadata
                processing_end = datetime.now()
                validated_analysis["processing"] = {
                    "status": "complete",
                    "pipeline_version": "2025-09-07-v1",
                    "attempts": 1,
                    "started_at": processing_start.isoformat(),
                    "completed_at": processing_end.isoformat(),
                    "error": None,
                    "logs": [
                        "Step 1: content sanitized and validated",
                        "Step 2: LLM analysis completed",
                        "Step 3: results validated against universal schema"
                    ]
                }
                
                return validated_analysis
                
            except ValueError as e:
                return self._get_fallback_analysis(f"JSON parsing error: {str(e)}")
            
        except Exception as e:
            print(f"Content analysis error: {str(e)}")
            return self._get_fallback_analysis(f"Analysis failed: {str(e)}")
    
    def _get_fallback_analysis(self, error_msg: str) -> dict:
        """Generate fallback analysis result with universal schema compliance"""
        return {
            "category": ["General"],
            "content_type": "Discussion",
            "complexity_level": "Intermediate", 
            "language": "en",
            "key_topics": ["general-content"],
            "named_entities": [],
            "processing": {
                "status": "failed",
                "pipeline_version": "2025-09-07-v1",
                "attempts": 1,
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "error": error_msg,
                "logs": [f"Error: {error_msg}"]
            }
        }
    
    async def process_video(self, youtube_url: str, summary_type: str = "comprehensive", proficiency_level: str = None) -> Dict:
        """Complete processing pipeline for a YouTube video
        
        Args:
            youtube_url: YouTube video URL
            summary_type: Type of summary to generate
            
        Returns:
            Complete analysis dictionary
        """
        print(f"Processing video: {youtube_url}")
        
        # Step 1: Extract transcript
        print("Extracting transcript...")
        transcript_data = self.extract_transcript(youtube_url)
        
        if not transcript_data.get('success'):
            return {
                'error': transcript_data.get('error'),
                'url': youtube_url,
                'processed_at': datetime.now().isoformat()
            }
        
        metadata = transcript_data['metadata']
        transcript_language = transcript_data.get('transcript_language') or metadata.get('language') or 'en'
        metadata['language'] = transcript_language
        transcript = transcript_data['transcript']
        content_type = transcript_data.get('content_type', 'transcript')
        
        # REJECT only videos with truly no content (very rare now with hybrid system)
        if content_type == 'none' or len(transcript.strip()) < 50:
            print("🚫 REJECTED: Cannot generate summary - no usable content available")
            return {
                'error': 'No usable content available for this video. Neither transcript nor description could be extracted. This may be a private video, age-restricted content, or have other access restrictions.',
                'url': youtube_url,
                'metadata': metadata,
                'transcript_available': False,
                'processed_at': datetime.now().isoformat(),
                'suggestion': "Try a different video or check if this video is publicly accessible."
            }
        elif content_type == 'description':
            # Allow description-only but warn the user
            print("⚠️  WARNING: Only video description available - summary will be limited")
            # Continue processing but mark as limited accuracy
        
        if len(transcript.strip()) < 50:
            print(f"⚠️  Transcript extracted: {len(transcript)} characters (insufficient for processing)")
        else:
            print(f"✅ Transcript extracted: {len(transcript)} characters")
        
        # Check if transcript is too short to be useful
        if len(transcript.strip()) < 50:  # Less than 50 characters is likely unusable
            video_title = metadata.get('title', 'Unknown video')
            return {
                'error': f"❌ No usable transcript found for '{video_title}'. This video appears to have no captions or subtitles available. Please try a different video that includes captions/subtitles for best results.",
                'url': youtube_url,
                'metadata': metadata,
                'transcript_length': len(transcript),
                'processed_at': datetime.now().isoformat(),
                'suggestion': "Look for videos with the [CC] closed captions icon on YouTube, or videos from channels that typically include subtitles."
            }
        
        # Step 2: Generate summary
        print("Generating summary...")
        summary_data = await self.generate_summary(transcript, metadata, summary_type, proficiency_level)

        summary_language = None
        if isinstance(summary_data, dict):
            summary_language = summary_data.get('language')
            if 'summary' not in summary_data and isinstance(summary_data.get('audio'), str):
                summary_data['summary'] = summary_data['audio']

        if isinstance(summary_data, str) and not summary_language:
            summary_language = transcript_language
        
        # Step 3: Analyze content (using summary for token efficiency)
        print("Analyzing content...")
        if isinstance(summary_data, dict):
            summary_text = summary_data.get('summary', '') or ''
            if not summary_text:
                for key in ('comprehensive', 'key_insights', 'bullet_points', 'audio'):
                    candidate = summary_data.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        summary_text = candidate.strip()
                        break
        else:
            summary_text = str(summary_data)

        if not summary_text.strip():
            print("⚠️ Summary text unavailable for analysis. Falling back to transcript excerpt.")
            summary_text = transcript[:6000]

        analysis_data = await self.analyze_content(summary_text, metadata)
        if summary_language:
            analysis_data['language'] = summary_language
        else:
            summary_language = analysis_data.get('language', transcript_language)
        
        # Enhanced universal schema compliance
        from urllib.parse import urlparse
        parsed_url = urlparse(youtube_url)
        video_id = metadata.get('id', metadata.get('video_id', ''))[:20]  # Limit ID length
        
        # Create universal schema structure
        result = {
            'id': f"yt:{video_id}" if video_id else f"yt:unknown-{int(datetime.now().timestamp())}",
            'content_source': 'youtube',
            'title': metadata.get('title', '')[:300],  # Limit title length
            'canonical_url': youtube_url,
            'thumbnail_url': metadata.get('thumbnail', ''),
            'published_at': self._format_youtube_date(metadata.get('upload_date', '')),
            'duration_seconds': metadata.get('duration', 0),
            'word_count': len(transcript.split()) if transcript else 0,
            
            'media': {
                'has_audio': True,  # YouTube videos always have audio
                'audio_duration_seconds': metadata.get('duration', 0),
                'has_transcript': bool(transcript and len(transcript.strip()) > 50),
                'transcript_chars': len(transcript) if transcript else 0
            },
            
            'source_metadata': {
                'youtube': {
                    'video_id': video_id,
                    'channel_name': metadata.get('uploader', 'Unknown')[:100],
                    'view_count': metadata.get('view_count', 0),
                    'tags': (metadata.get('tags', []) or [])[:10]  # Limit tags
                }
            },
            
            'analysis': analysis_data,
            'original_language': transcript_language,
            'summary_language': summary_language,
            'audio_language': summary_language,
            
            # Legacy fields for backward compatibility
            'url': youtube_url,
            'metadata': metadata,
            'transcript': transcript,
            'transcript_segments': transcript_segments,
            'summary': summary_data,
            'processed_at': datetime.now().isoformat(),
            'processor_info': {
                'llm_provider': self.llm_provider,
                'model': getattr(self.llm, 'model_name', getattr(self.llm, 'model', self.model))
            }
        }
        
        print("✅ Processing complete with universal schema compliance!")
        return result
    
    def _format_youtube_date(self, upload_date: str) -> str:
        """Convert YouTube upload_date (YYYYMMDD) to ISO 8601 format"""
        if not upload_date or len(upload_date) != 8:
            return datetime.now().isoformat()
        
        try:
            dt = datetime.strptime(upload_date, '%Y%m%d')
            return dt.isoformat() + 'Z'
        except ValueError:
            return datetime.now().isoformat() + 'Z'

    async def _generate_chunked_summary(self, transcript: str, metadata: Dict, summary_type: str, proficiency_level: str = None) -> Dict[str, str]:
        """Generate summary by processing transcript in chunks and combining results"""
        
        # Normalize summary_type to canonical keys for robust matching
        TYPE_MAP = {
            "comprehensive": "comprehensive",
            "key points": "key_points", 
            "key_points": "key_points",
            "bullet points": "key_points",
            "bullet-points": "key_points", 
            "insights": "insights",
            "key-insights": "insights",
            "key_insights": "insights",
            "audio summary": "audio",
            "audio-summary": "audio", 
            "audio": "audio",
            "audio-fr": "audio_fr",
            "audio-es": "audio_es",
        }
        normalized_type = TYPE_MAP.get(summary_type.lower().strip(), summary_type)
        
        # Split transcript into manageable chunks (aim for ~8000 chars per chunk with overlap)
        chunk_size = 8000
        overlap = 500  # Overlap between chunks to maintain context
        
        chunks = []
        start = 0
        while start < len(transcript):
            end = start + chunk_size
            if end < len(transcript):
                # Find a good breaking point (sentence end) within the overlap zone
                overlap_start = max(start, end - overlap)
                break_point = transcript.rfind('. ', overlap_start, end)
                if break_point > overlap_start:
                    end = break_point + 1
            
            chunk = transcript[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        print(f"🔄 Processing transcript in {len(chunks)} chunks...")
        
        # Process each chunk and collect summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"📝 Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)")
            
            # Create context for this chunk
            chunk_context = f"""
            Video Title: {metadata.get('title', 'Unknown')}
            Channel: {metadata.get('uploader', 'Unknown')}
            Upload Date (YYYYMMDD): {metadata.get('upload_date', 'Unknown')}
            Duration (seconds): {metadata.get('duration', 0)}
            URL: {metadata.get('url', 'Unknown')}
            
            Transcript Segment {i+1}/{len(chunks)}:
            {chunk}
            """
            
            # Generate focused summary for this chunk
            chunk_prompt = f"""
            {chunk_context}

            Summarize this transcript segment for later merging.

            Output:
            - 5–9 “• ” bullets with concrete facts, names, and numbers.
            - Avoid intro/outro fluff and repetition.
            - No headings, no meta; just bullets.
            - Use the transcript’s language.

            Keep bullets ≤ 18 words. Timestamps only if explicitly present in this segment.
            """
            
            chunk_summary = await self._robust_llm_call([HumanMessage(content=chunk_prompt)], 
                                                      operation_name="chunk summary generation")
            if chunk_summary:
                chunk_summaries.append(f"**Section {i+1}:**\n{chunk_summary}")
            else:
                chunk_summaries.append(f"**Section {i+1}:** [Processing failed for this section]")
        
        # Combine all chunk summaries into final summary
        print(f"🔄 Combining {len(chunk_summaries)} section summaries...")
        combined_summary = await self._combine_chunk_summaries(chunk_summaries, metadata, summary_type)

        # Only generate the requested summary type to save processing time and API costs
        result = {"comprehensive": combined_summary}

        if normalized_type == "key_points":
            result["bullet_points"] = await self._extract_bullet_points(combined_summary)
        elif normalized_type == "insights": 
            result["key_insights"] = await self._extract_key_insights(combined_summary)
        elif normalized_type == "audio":
            result["audio"] = await self._create_audio_summary(combined_summary)
        elif normalized_type == "audio_fr":
            result["audio"] = await self._create_translated_audio_summary(combined_summary, "French", proficiency_level)
        elif normalized_type == "audio_es":
            result["audio"] = await self._create_translated_audio_summary(combined_summary, "Spanish", proficiency_level)
        # For "comprehensive" type, we already have the comprehensive summary

        variant_map = {
            "comprehensive": "comprehensive",
            "key_points": "key-points",
            "insights": "key-insights",
            "audio": "audio",
            "audio_fr": "audio-fr",
            "audio_es": "audio-es",
        }

        primary_variant_key_map = {
            "key_points": "bullet_points",
            "insights": "key_insights",
            "audio": "audio",
            "audio_fr": "audio",
            "audio_es": "audio",
        }

        primary_key = primary_variant_key_map.get(normalized_type, "comprehensive")
        primary_text = result.get(primary_key) or combined_summary or ""

        canonical_variant = variant_map.get(normalized_type, "comprehensive")
        result["summary"] = primary_text
        result["summary_type"] = canonical_variant
        result["generated_at"] = datetime.now().isoformat()
        if normalized_type == "audio_fr":
            result["language"] = 'fr'
        elif normalized_type == "audio_es":
            result["language"] = 'es'
        else:
            result["language"] = metadata.get('language', 'en')

        if primary_text:
            title_prompt = f"""
            Write a single, specific headline (12–16 words, no emojis, no colon) that states subject and concrete value.
            **IMPORTANT: Respond in the same language as the content.**
            Source title: {metadata.get('title', '')}
            Summary:
            {primary_text[:1200]}
            """

            headline_text = await self._robust_llm_call(
                [HumanMessage(content=title_prompt)],
                "Headline generation"
            )

            result['headline'] = headline_text or "Generated Summary"
            print(f"📝 Generated Headline: {result['headline']}")

        return result

    async def _combine_chunk_summaries(self, chunk_summaries: List[str], metadata: Dict, summary_type: str) -> str:
        """Combine individual chunk summaries into a cohesive final summary"""
        
        combined_content = "\n\n".join(chunk_summaries)
        
        combine_prompt = f"""
        You have been provided with {len(chunk_summaries)} section summaries from a video transcript. 
        Your task is to combine these into one comprehensive, well-structured summary.
        
        Video Context:
        Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('uploader', 'Unknown')}
        Duration: {metadata.get('duration', 0)} seconds
        
        Section Summaries:
        {combined_content}
        
        Create a comprehensive final summary that:
        1. Integrates all key points from all sections into a cohesive narrative
        2. Identifies and connects main themes across the entire video
        3. Removes redundancy while preserving important details
        4. Organizes information logically with clear sections
        5. Provides actionable insights and key takeaways
        6. Maintains appropriate length and detail level
        
        Structure your response with clear headings and bullet points where appropriate.
        """
        
        final_summary = await self._robust_llm_call([HumanMessage(content=combine_prompt)], 
                                                  operation_name="final summary combination")
        
        return final_summary or "Unable to generate combined summary"

    async def _extract_bullet_points(self, summary: str) -> str:
        """Extract key points in bullet format"""
        prompt = f"""
        Convert this summary into a skim‑ready bullet list.

        {summary}

        Output:
        - One sentence overview.
        - 10–16 “• ” bullets with concrete facts, results, and proper names.
        - Finish with “Bottom line: …”.

        Rules:
        - Each bullet ≤ 18 words; lead with the key fact/action.
        - Prefer specifics (metrics, versions, dates). Merge near‑duplicates.
        - No headings, no code fences, no emojis.
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="bullet points extraction")
        return result or "Unable to generate bullet points"

    async def _extract_key_insights(self, summary: str) -> str:
        """Organize summary into content-derived categories with concise bullets"""
        prompt = f"""
        Organize the following summary into thematic categories based on the content itself (not external analysis).

        {summary}

        Output:
        - 3–5 short category headings derived from the content.
        - Under each heading, include 3–5 “• ” bullets capturing concrete facts, results, names, or metrics.

        Rules:
        - Keep each bullet ≤ 18 words; lead with the fact/action; avoid duplication.
        - No speculation; use “Unknown” only when details are missing.
        - No code fences or emojis.
        - Do not add a final “Bottom line”.
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="key insights extraction")
        return result or "Unable to generate categorized insights"

    async def _create_audio_summary_from_transcript(self, transcript: str, metadata: Dict) -> str:
        """Create audio-optimized summary directly from transcript (for direct path)"""
        base_context = f"""
        Video Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('uploader', 'Unknown')}
        Upload Date (YYYYMMDD): {metadata.get('upload_date', 'Unknown')}
        Duration (seconds): {metadata.get('duration', 0)}
        URL: {metadata.get('url', 'Unknown')}

        Full Transcript:
        {transcript}
        """
        
        prompt = f"""
        {base_context}

        Write a natural, text‑to‑speech friendly narration from the transcript.

        Structure:
        - Opening: 2–3 sentences that jump straight to the substance and key conclusion.
        - Main: Smoothly connect major topics with conversational transitions (no headings/bullets).
                 Use transitions like “First…”, “Next…”, “However…”, “The key trade‑off is…”.
        - Closing: One sentence starting “Bottom line: …”.

        Requirements:
        - Flowing paragraphs only; no lists, bullets, headings, code fences, or emojis.
        - Present findings directly; avoid meta like “the video shows…”.
        - Keep names/numbers accurate; include specific values where present.
        - Maintain an engaging pace with varied sentence lengths.
        - **IMPORTANT: Respond in the same language as the original transcript.** 

        Length guidance (not strict): 180–380 words for most videos; shorter for very short clips.
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="audio summary from transcript")
        return result or f"Unable to create audio summary from transcript"

    async def _create_audio_summary(self, summary: str) -> str:
        """Create audio-optimized version of summary"""
        prompt = f"""
        Convert this summary into a natural, text‑to‑speech friendly narration.

        {summary}

        Rules:
        - Use flowing paragraphs only (no bullets/headings/code fences/emojis).
        - Smooth transitions between topics; keep proper nouns/numbers accurate.
        - Maintain all key information and conclusions.
        - End with a single sentence beginning “Bottom line: …”.
        - Length target: 180–350 words (shorter for very short content).
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="audio summary creation")
        return result or summary  # Fallback to original summary

    async def _create_translated_audio_summary(self, summary: str, target_language: str, proficiency_level: str = None) -> str:
        """Create audio-optimized summary translated to target language with proficiency-based vocabulary support"""
        
        # Map proficiency levels to learning-friendly settings
        level_map = {
            "beginner": {
                "label": {"French": "Débutant", "Spanish": "Principiante"},
                "cefr": "A2–B1",
                "style": "phrases courtes et simples; vocabulaire de base; rythme lent; explications claires des termes techniques.",
                "vocab_items": 8,
            },
            "intermediate": {
                "label": {"French": "Intermédiaire", "Spanish": "Intermedio"},
                "cefr": "B1–B2",
                "style": "phrases bien structurées; vitesse modérée; quelques explications des termes spécialisés.",
                "vocab_items": 6,
            },
            "advanced": {
                "label": {"French": "Avancé", "Spanish": "Avanzado"},
                "cefr": "C1",
                "style": "registre naturel et fluide; rythme natif; peu d'explications nécessaires.",
                "vocab_items": 0,  # no dedicated vocab section
            },
        }
        
        prof = level_map.get((proficiency_level or "advanced").lower())
        logging.info(f"🎓 DEBUG: proficiency_level='{proficiency_level}', prof={prof}")
        
        # Build vocabulary warm-up section if needed
        vocab_block_req = ""
        if prof and prof["vocab_items"] > 0:
            if target_language == "French":
                vocab_block_req = f"""
                **Commencez par un section "Vocabulaire" (≈{prof['vocab_items']} termes):**
                - Sélectionnez les mots/expressions techniques ou peu fréquents du résumé
                - Format: **Mot** — définition simple (1 ligne) + phrase d'exemple courte
                - Présentez en français naturel pour faciliter l'apprentissage
                
                """
                logging.info(f"🎓 DEBUG: vocab_block_req set for French: {bool(vocab_block_req)}")
            else:  # Spanish
                vocab_block_req = f"""
                **Comience con una sección "Glosario" (≈{prof['vocab_items']} términos):**
                - Seleccione palabras/expresiones técnicas o poco frecuentes del resumen
                - Formato: **Palabra** — definición sencilla (1 línea) + frase de ejemplo corta
                - Presente en español natural para facilitar el aprendizaje
                
                """

        style_hint = prof["style"] if prof else "tono conversacional natural; fluidez nativa."
        level_tag = prof["label"][target_language] if prof else ("Avancé" if target_language == "French" else "Avanzado")

        # Determine if this is translation or language-level adaptation
        instruction_verb = "Adaptez" if "french" in summary.lower() and target_language == "French" else \
                          "Adapte" if "español" in summary.lower() and target_language == "Spanish" else \
                          f"Traduisez en {target_language} et adaptez"

        prompt = f"""
        {instruction_verb} ce résumé pour la synthèse vocale en **{target_language}** (niveau: {level_tag}):
        
        {summary}
        
        **Exigences:**
        - Niveau de langue: {style_hint}
        - Maintenez toute l'information clé avec précision technique
        - Transitions fluides entre les idées; évitez les listes à puces
        - Conservez les noms propres et chiffres avec format local approprié
        - Ton engageant et informatif, adapté à l'écoute
        - **Même si le contenu source est déjà en {target_language}, ajustez le niveau de complexité et ajoutez le vocabulaire selon les spécifications**
        
        {vocab_block_req}
        
        **Structure de sortie:**
        1) {f"Section Vocabulaire/Glosario (si demandée ci-dessus)" if vocab_block_req else ""}
        2) **Narration principale**: texte prêt pour la synthèse vocale en {target_language}, adapté au niveau {level_tag},
           se terminant par une phrase commençant par « Bottom line: … » dans la langue cible.

        **Répondez uniquement en {target_language}.**
        """
        
        logging.info(f"🎓 DEBUG: Final prompt has vocab_block_req: {bool(vocab_block_req)}")
        logging.info(f"🎓 DEBUG: Prompt preview: {prompt[:300]}...")
        
        result = await self._robust_llm_call(
            [HumanMessage(content=prompt)], 
            operation_name=f"translated audio summary ({target_language}/{proficiency_level or 'advanced'})"
        )
        return result or f"Unable to create {target_language} audio summary"

    async def _condense_for_tts(self, text: str) -> str:
        """Condense a long summary into TTS-friendly length while preserving key content"""
        # Target must be under 4096 - use 3900 as safe target
        target_chars = 3900
        target_words = 550  # Roughly 3900 chars at ~7 chars per word
        
        condense_prompt = f"""
        CRITICAL: You must condense this {len(text)}-character summary to EXACTLY {target_words} words (approximately {target_chars} characters) for TTS compatibility.

        The current summary is {len(text)} characters and EXCEEDS the 4096 character limit. You must make significant cuts while preserving the most important content.

        PRESERVE IN ORDER OF IMPORTANCE:
        1. Main conclusions and final recommendations
        2. Key specific findings and comparisons
        3. Important numbers, prices, measurements
        4. Critical trade-offs and practical implications

        AGGRESSIVE CUTS REQUIRED:
        - Combine multiple sentences into single, direct statements  
        - Eliminate all redundant explanations and examples
        - Remove transition phrases and filler language
        - Merge similar points together
        - Cut lengthy descriptions down to essential facts
        - Use shorter, more direct wording throughout

        AIM FOR EXACTLY {target_words} WORDS. This is NOT optional - the summary must fit the TTS limit.

        Original summary:
        {text}
        """
        
        condensed_content = await self._robust_llm_call(
            [HumanMessage(content=condense_prompt)], 
            "Summary condensation"
        )
        
        if condensed_content:
            return condensed_content
        else:
            print("⚠️ Failed to condense summary for TTS, using truncated version")
            # Fallback: intelligently truncate at sentence boundary
            if len(text) <= 3800:
                return text
            sentences = text.split('. ')
            condensed = ""
            for sentence in sentences:
                if len(condensed + sentence + ". ") <= 3800:
                    condensed += sentence + ". "
                else:
                    break
            return condensed.strip()
    
    def _split_text_for_tts(self, text: str, max_chunk_chars: int = 3800) -> List[str]:
        """Split text into TTS-friendly chunks at sentence boundaries"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            # Add period back unless it's the last sentence
            sentence_with_period = sentence + '.' if not sentence.endswith('.') else sentence
            
            # Check if adding this sentence would exceed limit
            if len(current_chunk) + len(sentence_with_period) + 1 > max_chunk_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_with_period
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence_with_period
                else:
                    current_chunk = sentence_with_period
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    async def _generate_chunked_tts(
        self,
        text_chunks: List[str],
        base_filename: str,
        json_filepath: str = None,
        *,
        provider: str = "openai",
        local_client: Optional[TTSHubClient] = None,
        voice: Optional[str] = None,
        engine: Optional[str] = None,
        favorite_slug: Optional[str] = None,
        progress: Optional[object] = None,
    ) -> Optional[str]:
        """Generate TTS for multiple chunks and combine them"""
        import tempfile
        import shutil
        
        async def _call_progress(event: dict):
            if not progress:
                return
            try:
                res = progress(event)
                import asyncio as _asyncio
                if _asyncio.iscoroutine(res):
                    await res  # type: ignore
            except Exception:
                pass

        chunk_files = []
        
        try:
            # Generate TTS for each chunk
            for i, chunk in enumerate(text_chunks):
                print(f"🎵 Generating TTS for chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                try:
                    await _call_progress({'stage': 'chunk_start', 'index': i+1, 'total': len(text_chunks), 'chars': len(chunk)})
                except Exception:
                    pass
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    chunk_filename = temp_file.name
                
                # Generate TTS for this chunk
                chunk_result = await self._generate_single_tts(
                    chunk,
                    chunk_filename,
                    provider=provider,
                    local_client=local_client,
                    voice=voice,
                    engine=engine,
                    favorite_slug=favorite_slug,
                )
                if chunk_result:
                    chunk_files.append(chunk_filename)
                    print(f"✅ Generated chunk {i+1}")
                    try:
                        await _call_progress({'stage': 'chunk_done', 'index': i+1, 'total': len(text_chunks)})
                    except Exception:
                        pass
                else:
                    print(f"❌ Failed to generate chunk {i+1}")
                    try:
                        await _call_progress({'stage': 'chunk_fail', 'index': i+1, 'total': len(text_chunks)})
                    except Exception:
                        pass
                    # Clean up any files created so far
                    for f in chunk_files:
                        try:
                            import os
                            os.unlink(f)
                        except:
                            pass
                    return None
            
            # Combine audio files
            if not base_filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_filename = f"combined_tts_{timestamp}.mp3"
            
            # Ensure exports directory exists and get full path
            exports_dir = Path('exports')
            exports_dir.mkdir(exist_ok=True)
            final_filename = str(exports_dir / base_filename)
            
            try:
                await _call_progress({'stage': 'combining', 'parts': len(chunk_files)})
            except Exception:
                pass
            if len(chunk_files) == 1:
                # Only one chunk, just rename it
                shutil.move(chunk_files[0], final_filename)
            else:
                # Try to combine with ffmpeg, fall back to simple concatenation
                combined = self._combine_audio_files(chunk_files, final_filename)
                if not combined:
                    print("⚠️ Audio combination failed, using first chunk only")
                    shutil.move(chunk_files[0], final_filename)
            
            print(f"✅ Combined TTS audio saved: {final_filename}")
            try:
                await _call_progress({'stage': 'combined', 'path': final_filename})
            except Exception:
                pass
            
            # Update JSON with MP3 metadata if generation succeeded
            if json_filepath:
                self._update_json_with_mp3_metadata(json_filepath, final_filename)
            
            try:
                await _call_progress({'stage': 'done', 'path': final_filename})
            except Exception:
                pass
            return final_filename
            
        except Exception as e:
            print(f"❌ Chunked TTS generation failed: {e}")
            try:
                await _call_progress({'stage': 'error', 'message': str(e)})
            except Exception:
                pass
            return None
        finally:
            # Clean up temporary chunk files
            for chunk_file in chunk_files:
                try:
                    import os
                    os.unlink(chunk_file)
                except:
                    pass
    
    async def _generate_single_tts(
        self,
        text: str,
        output_filename: str,
        *,
        provider: str = "openai",
        local_client: Optional[TTSHubClient] = None,
        voice: Optional[str] = None,
        engine: Optional[str] = None,
        favorite_slug: Optional[str] = None,
        progress: Optional[object] = None,
    ) -> Optional[str]:
        """Generate TTS for a single text chunk"""
        provider = (provider or "openai").lower()
        async def _call_progress(event: dict):
            if not progress:
                return
            try:
                res = progress(event)
                import asyncio as _asyncio
                if _asyncio.iscoroutine(res):
                    await res  # type: ignore
            except Exception:
                pass

        try:
            await _call_progress({'stage': 'single_start'})
        except Exception:
            pass

        if provider == "local":
            client = local_client or self._resolve_local_tts_client()
            if not client:
                raise LocalTTSUnavailable("Local TTS hub is not configured.")
            try:
                import logging as _logging
                _logging.info(
                    "[TTS] local synth start base=%s engine=%s fav=%s voice=%s text_len=%d",
                    getattr(client, 'base_api_url', None),
                    engine,
                    favorite_slug,
                    voice,
                    len(text or ""),
                )
            except Exception:
                pass
            try:
                result = await client.synthesise(
                    text,
                    voice_id=voice,
                    engine=engine,
                    favorite_slug=favorite_slug,
                )
            except LocalTTSUnavailable:
                raise
            except Exception as exc:
                raise LocalTTSUnavailable(str(exc))

            audio_bytes = result.get("audio_bytes")
            if not audio_bytes:
                raise LocalTTSUnavailable("Local TTS hub returned no audio data.")
            with open(output_filename, "wb") as f:
                f.write(audio_bytes)
            try:
                await _call_progress({'stage': 'single_done', 'path': output_filename})
            except Exception:
                pass
            return output_filename

        # Default to OpenAI provider
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            try:
                await _call_progress({'stage': 'error', 'message': 'openai_key_missing'})
            except Exception:
                pass
            return None
            
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice or "fable",
            "response_format": "mp3"
        }
        
        response = self._make_request_with_retry(url, headers, payload)
        if response and response.status_code == 200:
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            return output_filename
        else:
            return None
    
    def _combine_audio_files(self, chunk_files: List[str], output_filename: str) -> bool:
        """Combine multiple audio files into one using ffmpeg if available"""
        try:
            import subprocess
            
            # Create a text file listing all input files
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{chunk_file}'\n")
                filelist_path = f.name
            
            # Use ffmpeg to concatenate
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filelist_path, '-c', 'copy', output_filename, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up
            import os
            os.unlink(filelist_path)
            
            if result.returncode == 0:
                print("✅ Audio files combined successfully with ffmpeg")
                return True
            else:
                print(f"⚠️ ffmpeg failed: {result.stderr}")
                return False
                
        except (FileNotFoundError, subprocess.SubprocessError):
            print("⚠️ ffmpeg not available, cannot combine audio chunks")
            return False
        except Exception as e:
            print(f"⚠️ Audio combination error: {e}")
            return False

    def _make_request_with_retry(self, url: str, headers: dict, payload: dict, max_retries: int = 3, timeout: int = 60) -> Optional[object]:
        """Make HTTP request with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 TTS API attempt {attempt + 1}/{max_retries}")
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ API Error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:  # Last attempt
                        return response
                    time.sleep(2 ** attempt)  # Wait before retry
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except Exception as e:
                print(f"⚠️ Request error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None

    async def _robust_llm_call(self, messages: list, operation_name: str = "LLM call", max_retries: int = 3) -> Optional[str]:
        """Make LLM API call with timeout and retry logic"""
        # Prefer local client (hub proxy or direct Ollama) when provider=ollama and any local base is configured
        use_hub = False
        use_local_client = False
        if self.llm_provider == "ollama":
            if os.getenv("TTSHUB_API_BASE"):
                use_hub = True
                use_local_client = True
            elif os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST"):
                use_local_client = True
        for attempt in range(max_retries):
            try:
                print(f"🔄 {operation_name} attempt {attempt + 1}/{max_retries}")
                if use_local_client:
                    # Fast-fail if local hub/direct is not reachable
                    if not self._ollama_precheck(timeout=0.8):
                        raise OllamaClientError("ollama precheck failed (unreachable)")
                    # Route through our unified local client (hub or direct)
                    txt = await asyncio.wait_for(self._ollama_hub_call(messages, max_retries=1), timeout=30.0)
                    return txt
                else:
                    # Use asyncio.wait_for to implement timeout
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(messages),
                        timeout=120.0  # 2 minute timeout for LLM calls
                    )
                    return response.content
                
            except asyncio.TimeoutError:
                print(f"⚠️ {operation_name} timed out on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                continue
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str:
                    wait_time = (2 ** attempt) + 5  # Longer wait for rate limits
                    print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"⚠️ {operation_name} error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
        
        print(f"❌ All {operation_name} attempts failed")
        if use_local_client:
            # Signal to the caller so it can fall back to cloud summarizer
            raise OllamaClientError(f"{operation_name} failed via local hub after retries")
        return None

    async def generate_tts_audio(
        self,
        text: str,
        output_filename: str = None,
        json_filepath: str = None,
        *,
        provider: str = "openai",
        voice: Optional[str] = None,
        engine: Optional[str] = None,
        favorite_slug: Optional[str] = None,
        progress: Optional[object] = None,
    ) -> Optional[str]:
        """Generate TTS audio with support for local hub or OpenAI."""
        provider = (provider or "openai").lower()
        try:
            import logging as _logging
            _logging.info("[TTS] enter generate_tts_audio provider=%s text_len=%d", provider, len(text or ""))
        except Exception:
            pass

        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"audio_summary_{timestamp}.mp3"

        # Ensure exports directory exists
        exports_dir = Path('exports')
        exports_dir.mkdir(exist_ok=True)
        output_path = exports_dir / output_filename

        # Determine whether chunking is required (provider-aware)
        # OpenAI typically needs ~4090 char chunks; local hubs may benefit from smaller chunks.
        local_chunk_chars = None
        try:
            local_chunk_chars = int(os.getenv('TTS_LOCAL_CHUNK_CHARS', '1800'))
        except Exception:
            local_chunk_chars = 1800
        openai_chunk_chars = None
        try:
            openai_chunk_chars = int(os.getenv('TTS_OPENAI_CHUNK_CHARS', '4090'))
        except Exception:
            openai_chunk_chars = 4090

        if provider == 'local':
            limit = max(500, local_chunk_chars or 1800)
        else:
            limit = max(1000, openai_chunk_chars or 4090)

        should_chunk = len(text) > limit
        chunks = self._split_text_for_tts(text, max_chunk_chars=limit) if should_chunk else [text]
        try:
            import logging as _logging
            _logging.info("[TTS] chunk decision provider=%s limit=%d chunks=%d", provider, limit, len(chunks))
        except Exception:
            pass

        async def _call_progress(event: dict):
            if not progress:
                return
            try:
                res = progress(event)
                import asyncio as _asyncio
                if _asyncio.iscoroutine(res):
                    await res  # type: ignore
            except Exception:
                pass

        local_client: Optional[TTSHubClient] = None
        if provider == "local":
            local_client = self._resolve_local_tts_client()
            if not local_client:
                raise LocalTTSUnavailable("Local TTS hub is not configured.")

        try:
            # Initial progress signal
            await _call_progress({
                'stage': 'init',
                'provider': provider,
                'text_len': len(text),
                'chunks': len(chunks),
            })
            if len(chunks) > 1:
                print(f"📝 Audio summary is {len(text)} characters, generating across {len(chunks)} chunks…")
                result_path = await self._generate_chunked_tts(
                    chunks,
                    output_filename,
                    json_filepath,
                    provider=provider,
                    local_client=local_client,
                    voice=voice,
                    engine=engine,
                    favorite_slug=favorite_slug,
                    progress=progress,
                )
            else:
                result_path = await self._generate_single_tts(
                    chunks[0],
                    str(output_path),
                    provider=provider,
                    local_client=local_client,
                    voice=voice,
                    engine=engine,
                    favorite_slug=favorite_slug,
                    progress=progress,
                )
                if result_path and json_filepath:
                    self._update_json_with_mp3_metadata(json_filepath, str(output_path))

            if result_path and Path(result_path).exists():
                print(f"✅ TTS audio saved to {result_path}")
                return result_path

            print("❌ TTS generation returned no audio")
            try:
                await _call_progress({'stage': 'error', 'message': 'no_audio'})
            except Exception:
                pass
            return None

        except LocalTTSUnavailable as exc:
            print(f"⚠️ Local TTS unavailable: {exc}")
            raise
        except Exception as exc:
            print(f"❌ TTS generation failed: {exc}")
            try:
                await _call_progress({'stage': 'error', 'message': str(exc)})
            except Exception:
                pass
            return None

    def _update_json_with_mp3_metadata(self, json_filepath: str, mp3_filepath: str) -> bool:
        """Update JSON report with MP3 metadata after TTS generation

        Args:
            json_filepath: Path to the JSON report file to update
            mp3_filepath: Path to the generated MP3 file
            
        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            # Import the update function from report_generator
            from modules.report_generator import update_json_with_mp3_metadata
            
            # Extract voice from the TTS settings (hardcoded to "fable" for now)
            voice = "fable"  # This matches the voice used in generate_tts_audio
            
            # Update the JSON file with MP3 metadata
            success = update_json_with_mp3_metadata(json_filepath, mp3_filepath, voice)
            
            if success:
                print(f"✅ Updated JSON with MP3 metadata: {json_filepath}")
            else:
                print(f"❌ Failed to update JSON with MP3 metadata: {json_filepath}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error updating JSON with MP3 metadata: {e}")
            return False

    def _resolve_local_tts_client(self) -> Optional[TTSHubClient]:
        if self._tts_hub_client and self._tts_hub_client.base_api_url:
            return self._tts_hub_client
        client = TTSHubClient.from_env()
        if client and client.base_api_url:
            self._tts_hub_client = client
            return client
        return None

        # HUME AI TTS (COMMENTED OUT - keeping for reference)
        # try:
        #     # Get API key from environment
        #     api_key = os.getenv('HUME_API_KEY')
        #     if not api_key:
        #         print("❌ HUME_API_KEY not found in environment")
        #         return None
        #     
        #     # Prepare API request
        #     url = "https://api.hume.ai/v0/tts/file"
        #     payload = {
        #         "utterances": [{
        #             "text": text
        #         }],
        #         "format": {"type": "mp3"}
        #     }
        #     headers = {
        #         "X-Hume-Api-Key": api_key,
        #         "Content-Type": "application/json"
        #     }
        #     
        #     print("🎙️ Generating TTS audio...")
        #     response = requests.post(url, headers=headers, json=payload, timeout=30)
        #     
        #     if response.status_code == 200:
        #         with open(output_path, "wb") as f:
        #             f.write(response.content)
        #         print(f"✅ TTS audio saved to {output_path}")
        #         return str(output_path)
        #     else:
        #         print(f"❌ TTS API Error: {response.status_code} - {response.text}")
        #         return None
        #         
        # except Exception as e:
        #     print(f"❌ TTS generation failed: {str(e)}")
        #     return None


async def main():
    """Example usage of the YouTube Summarizer"""
    
    # Initialize summarizer (requires API keys in .env file)
    try:
        summarizer = YouTubeSummarizer(llm_provider="openai", model="gpt-4")
    except Exception as e:
        print(f"Failed to initialize summarizer: {e}")
        print("Make sure you have OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return
    
    # Example YouTube URL (replace with actual video)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll as safe test
    
    # Process the video
    result = await summarizer.process_video(test_url, summary_type="comprehensive")
    
    # Display results
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("\n" + "="*60)
        print("YOUTUBE VIDEO SUMMARY")
        print("="*60)
        print(f"Title: {result['metadata']['title']}")
        print(f"Channel: {result['metadata']['uploader']}")
        print(f"Duration: {result['metadata']['duration']} seconds")
        print(f"Views: {result['metadata']['view_count']:,}")
        
        print(f"\nHeadline: {result['summary']['headline']}")
        print(f"\nSummary:\n{result['summary']['summary']}")
        
        print(f"\nContent Analysis:")
        analysis = result['analysis']
        print(f"Category: {', '.join(analysis.get('category', ['Unknown']))}")
        print(f"Sentiment: {analysis.get('sentiment', 'Unknown')}")
        print(f"Target Audience: {analysis.get('target_audience', 'Unknown')}")
        print(f"Complexity: {analysis.get('complexity_level', 'Unknown')}")
        print(f"Key Topics: {', '.join(analysis.get('key_topics', ['Unknown']))}")


if __name__ == "__main__":
    asyncio.run(main())

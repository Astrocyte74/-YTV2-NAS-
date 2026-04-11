"""
YTV2 HTTP API - Multi-channel gateway endpoints.

This FastAPI application provides internal HTTP endpoints for processing
YouTube videos, Reddit threads, and web pages. It's designed to be called
by multi-channel gateways like Clawdbot.

Security: Binds to 127.0.0.1 only - accessible only from within the NAS.
"""

from __future__ import annotations

import asyncio
import logging
import json
import os

logger = logging.getLogger(__name__)
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Header, Body, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .models import (
    ProcessRequest,
    ProcessResponse,
    TranscriptResponse,
    HealthResponse,
    FollowUpSuggestion,
    FollowUpSuggestionsRequest,
    FollowUpSuggestionsResponse,
    FollowUpRunRequest,
    FollowUpRunResponse,
    FollowUpCachedResponse,
    ResearchSource,
    FollowUpThreadResponse,
    FollowUpThreadTurn,
    FollowUpChatRequest,
    FollowUpChatResponse,
    FollowUpChatTurnResponse,
)

# Lazy-initialized service instances
_summarizer = None
_reddit_fetcher = None
_web_fetcher = None

# URL pattern matching
YOUTUBE_PATTERN = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[\w-]+'
)
REDDIT_PATTERN = re.compile(
    r'^(https?://)?(www\.)?(reddit\.com/|redd\.it/)[\w/]+'
)

# Research service constants
_RESEARCH_PATH_ADDED = False
_follow_up_store = None


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/live/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_dashboard_url() -> str:
    """Get the dashboard URL from environment."""
    return os.getenv('DASHBOARD_URL', os.getenv('POSTGRES_DASHBOARD_URL', ''))


def get_api_secret() -> Optional[str]:
    """Get the optional API secret for authentication."""
    return os.getenv('YTV2_API_SECRET')


def check_auth(authorization: Optional[str]) -> bool:
    """Check if the request is authorized (if secret is set)."""
    secret = get_api_secret()
    if not secret:
        # No auth required
        return True
    if not authorization:
        return False
    # Support both "Bearer <token>" and plain token
    token = authorization.replace('Bearer ', '').strip()
    return token == secret


async def require_auth(authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency for authentication - raises if auth fails."""
    if not check_auth(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for service initialization."""
    # Startup
    print("[YTV2 API] Starting YTV2 API server...")
    print(f"[YTV2 API] Dashboard URL: {get_dashboard_url() or 'Not configured'}")
    print(f"[YTV2 API] Auth required: {get_api_secret() is not None}")
    yield
    # Shutdown
    print("[YTV2 API] Shutting down...")


app = FastAPI(
    title="YTV2 Processing API",
    description="Internal API for multi-channel content processing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable docs in production
    redoc_url=None,
)


def get_summarizer():
    """Get or create the YouTubeSummarizer instance."""
    global _summarizer
    if _summarizer is None:
        from youtube_summarizer import YouTubeSummarizer
        _summarizer = YouTubeSummarizer()
    return _summarizer


def get_reddit_fetcher():
    """Get or create the RedditFetcher instance."""
    global _reddit_fetcher
    if _reddit_fetcher is None:
        from modules.sources.reddit import RedditFetcher
        try:
            _reddit_fetcher = RedditFetcher()
        except Exception as e:
            print(f"[YTV2 API] Reddit fetcher not available: {e}")
            _reddit_fetcher = None
    return _reddit_fetcher


def get_web_fetcher():
    """Get or create the WebPageFetcher instance."""
    global _web_fetcher
    if _web_fetcher is None:
        from modules.sources.web import WebPageFetcher
        _web_fetcher = WebPageFetcher()
    return _web_fetcher


async def process_youtube(url: str, summary_type: str, user_id: str, channel: str) -> ProcessResponse:
    """Process a YouTube URL."""
    summarizer = get_summarizer()

    try:
        result = await summarizer.process_video(url, summary_type)

        if result.get('error'):
            return ProcessResponse(
                content_id=result.get('video_id', 'unknown'),
                status='failed',
                error=result.get('error'),
                metadata=result.get('metadata', {}),
                dashboard_url=None,
                source_type='youtube'
            )

        video_id = result.get('video_id', 'unknown')
        metadata = result.get('metadata', {})
        dashboard_base = get_dashboard_url()

        # Extract summary from result
        summary_text = None
        if 'summary' in result:
            s = result['summary']
            # Check if summary is a string
            if isinstance(s, str):
                summary_text = s
            elif isinstance(s, dict):
                # Summary is a dict - extract the actual summary text
                if 'error' in s:
                    # Summary contains an error - return failed response
                    return ProcessResponse(
                        content_id=video_id,
                        status='failed',
                        error=s.get('error', 'Unknown error'),
                        metadata=metadata,
                        dashboard_url=None,
                        source_type='youtube'
                    )
                # Extract the summary text from the dict
                summary_text = s.get('summary') or s.get('headline')
                if summary_text and len(summary_text) > 1000:
                    # Truncate long summaries for API responses
                    summary_text = summary_text[:1000] + "..."
        elif 'summary_data' in result:
            summary_data = result['summary_data']
            # Try to get the requested summary type
            s = summary_data.get(summary_type) or summary_data.get('comprehensive') or summary_data.get('bullet-points')
            if isinstance(s, str):
                summary_text = s
            elif isinstance(s, dict):
                # Summary is a dict - extract text or html field
                summary_text = s.get('text') or s.get('html')
                if summary_text and len(summary_text) > 500:
                    # Truncate long summaries for API responses
                    summary_text = summary_text[:1000] + "..."

        return ProcessResponse(
            content_id=video_id,
            status='completed',
            summary=summary_text,
            metadata={
                'title': metadata.get('title', ''),
                'channel': metadata.get('channel', ''),
                'duration': metadata.get('duration', 0),
                'language': metadata.get('language', 'en'),
            },
            dashboard_url=f"{dashboard_base}/watch/{video_id}" if dashboard_base else None,
            source_type='youtube'
        )

    except Exception as e:
        return ProcessResponse(
            content_id='unknown',
            status='failed',
            error=str(e),
            metadata={},
            dashboard_url=None,
            source_type='youtube'
        )


async def process_reddit(url: str, summary_type: str, user_id: str, channel: str) -> ProcessResponse:
    """Process a Reddit URL."""
    fetcher = get_reddit_fetcher()
    if not fetcher:
        return ProcessResponse(
            content_id='unknown',
            status='failed',
            error='Reddit credentials not configured. Set REDDIT_CLIENT_ID, REDDIT_REFRESH_TOKEN, and REDDIT_USER_AGENT.',
            metadata={},
            dashboard_url=None,
            source_type='reddit'
        )

    try:
        reddit_result = fetcher.fetch(url)
        summarizer = get_summarizer()

        # Generate summary from combined text
        metadata = {
            'title': reddit_result.title,
            'url': url,
            'subreddit': reddit_result.subreddit,
            'author': reddit_result.author,
        }

        summary_text = await summarizer.generate_summary(
            reddit_result.combined_text,
            metadata
        )

        if isinstance(summary_text, dict):
            summary_text = summary_text.get('summary') or summary_text.get('comprehensive')

        # Ensure summary is a string, not a dict
        if isinstance(summary_text, dict):
            summary_text = str(summary_text)
        elif not isinstance(summary_text, str):
            summary_text = None

        dashboard_base = get_dashboard_url()

        return ProcessResponse(
            content_id=reddit_result.id,
            status='completed',
            summary=summary_text,
            metadata={
                'title': reddit_result.title,
                'subreddit': reddit_result.subreddit,
                'author': reddit_result.author,
                'score': reddit_result.score,
                'num_comments': reddit_result.num_comments,
                'language': reddit_result.language or 'en',
            },
            dashboard_url=f"{dashboard_base}/reddit/{reddit_result.id}" if dashboard_base else None,
            source_type='reddit'
        )

    except Exception as e:
        return ProcessResponse(
            content_id='unknown',
            status='failed',
            error=str(e),
            metadata={},
            dashboard_url=None,
            source_type='reddit'
        )


async def process_web(url: str, summary_type: str, user_id: str, channel: str) -> ProcessResponse:
    """Process a generic web page URL."""
    fetcher = get_web_fetcher()
    summarizer = get_summarizer()

    try:
        web_result = fetcher.fetch(url)
        content = web_result.content

        # Generate summary from extracted text
        metadata = {
            'title': content.title,
            'url': url,
            'site_name': content.site_name,
        }

        summary_text = await summarizer.generate_summary(
            web_result.text,
            metadata
        )

        if isinstance(summary_text, dict):
            summary_text = summary_text.get('summary') or summary_text.get('comprehensive')

        # Ensure summary is a string, not a dict
        if isinstance(summary_text, dict):
            summary_text = str(summary_text)
        elif not isinstance(summary_text, str):
            summary_text = None

        # Generate a simple ID from URL
        import hashlib
        content_id = hashlib.md5(url.encode()).hexdigest()[:12]

        return ProcessResponse(
            content_id=content_id,
            status='completed',
            summary=summary_text,
            metadata={
                'title': content.title,
                'site_name': content.site_name,
                'language': content.language or 'en',
                'url': url,
            },
            dashboard_url=None,  # Web content doesn't have dashboard pages
            source_type='web'
        )

    except Exception as e:
        return ProcessResponse(
            content_id='unknown',
            status='failed',
            error=str(e),
            metadata={},
            dashboard_url=None,
            source_type='web'
        )


async def get_youtube_transcript(url: str) -> TranscriptResponse:
    """Extract transcript from YouTube URL without summarization."""
    summarizer = get_summarizer()

    try:
        result = summarizer.extract_transcript(url)

        if result.get('error'):
            return TranscriptResponse(
                content_id=_extract_video_id(url) or 'unknown',
                status='failed',
                error=result.get('error'),
                metadata={},
                chapters=None,
                segments=None,
                language=None,
                duration=None,
                source_type='youtube'
            )

        metadata = result.get('metadata', {})

        return TranscriptResponse(
            content_id=_extract_video_id(url) or metadata.get('video_id', 'unknown'),
            status='completed',
            transcript=result.get('transcript'),
            metadata={
                'title': metadata.get('title', ''),
                'channel': metadata.get('channel', ''),
                'channel_id': metadata.get('channel_id', ''),
                'upload_date': metadata.get('upload_date', ''),
                'view_count': metadata.get('view_count', 0),
                'like_count': metadata.get('like_count', 0),
            },
            chapters=metadata.get('chapters', []),
            segments=result.get('transcript_segments', []),  # Timestamped transcript segments
            language=result.get('transcript_language'),  # Language of the transcript
            duration=metadata.get('duration'),
            source_type='youtube'
        )

    except Exception as e:
        return TranscriptResponse(
            content_id='unknown',
            status='failed',
            error=str(e),
            metadata={},
            chapters=None,
            segments=None,
            language=None,
            duration=None,
            source_type='youtube'
        )


@app.post("/api/process", response_model=ProcessResponse)
async def process_url(request: ProcessRequest, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    """
    Process a URL (YouTube, Reddit, or Web) and return summary.

    This endpoint accepts a URL, detects its type, and routes it to the
    appropriate processing pipeline. Results are returned directly in the
    response for quick delivery back to the user via any channel.
    """
    # Check authentication if secret is configured
    if not check_auth(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token"
        )

    url = request.url.strip()
    summary_type = request.summary_type or "comprehensive"
    user_id = request.user_id or "clawdbot"
    channel = request.channel or "unknown"

    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL is required"
        )

    # Detect URL type and route
    if YOUTUBE_PATTERN.match(url):
        return await process_youtube(url, summary_type, user_id, channel)
    elif REDDIT_PATTERN.match(url):
        return await process_reddit(url, summary_type, user_id, channel)
    else:
        return await process_web(url, summary_type, user_id, channel)


@app.post("/api/transcript", response_model=TranscriptResponse)
async def get_transcript(url: str = Body(..., embed=True), authorization: Optional[str] = Header(None)):
    """
    Extract transcript from a YouTube URL without summarization.

    This endpoint returns only the raw transcript and metadata, allowing the
    caller to perform their own summarization. No LLM costs are incurred.

    Use this when you want to:
    - Get the full transcript for custom processing
    - Let your own service handle summarization
    - Avoid LLM costs on the YTV2 side
    """
    # Check authentication if secret is configured
    if not check_auth(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token"
        )

    url = url.strip()

    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL is required"
        )

    # Only YouTube supported for transcript extraction
    if not YOUTUBE_PATTERN.match(url):
        return TranscriptResponse(
            content_id='unknown',
            status='failed',
            error='Only YouTube URLs are supported for transcript extraction',
            metadata={},
            chapters=None,
            segments=None,
            language=None,
            duration=None,
            source_type='youtube'
        )

    return await get_youtube_transcript(url)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="ytv2-api",
        version="1.0.0"
    )


# =============================================================================
# Follow-Up Research Endpoints
# =============================================================================

_research_service = None


def get_research_service():
    """Get or lazy-import the research service."""
    global _research_service, _RESEARCH_PATH_ADDED
    if _research_service is None:
        try:
            if not _RESEARCH_PATH_ADDED:
                research_api_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research_api')
                import sys
                if research_api_path not in sys.path:
                    sys.path.insert(0, research_api_path)
                _RESEARCH_PATH_ADDED = True

            from research_service.service import (
                get_follow_up_suggestions,
                run_follow_up_research,
                get_research_capabilities,
                answer_follow_up_chat,
                stream_report_chat,
            )
            from research_service.follow_up import build_cache_key
            _research_service = {
                'get_follow_up_suggestions': get_follow_up_suggestions,
                'run_follow_up_research': run_follow_up_research,
                'get_research_capabilities': get_research_capabilities,
                'build_cache_key': build_cache_key,
                'answer_follow_up_chat': answer_follow_up_chat,
                'stream_report_chat': stream_report_chat,
            }
            print("[YTV2 API] Research service loaded successfully")
        except Exception as e:
            print(f"[YTV2 API] Failed to load research service: {e}")
            _research_service = {'error': str(e)}
    return _research_service


def require_research_service():
    """Get research service or raise HTTPException if unavailable."""
    service = get_research_service()
    if 'error' in service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Research service not available: {service['error']}"
        )
    return service


def get_follow_up_store():
    """Get or lazy-import the follow-up persistence layer."""
    global _follow_up_store
    if _follow_up_store is None:
        from .follow_up_store import FollowUpStore
        _follow_up_store = FollowUpStore()
    return _follow_up_store


def _build_follow_up_run_response(
    *,
    video_id: str,
    summary_id: Optional[int],
    status: str,
    answer: str,
    sources: list,
    meta: Dict[str, Any],
    error: Optional[str] = None,
) -> FollowUpRunResponse:
    api_sources = [
        ResearchSource(
            name=src.name if hasattr(src, "name") else src.get("name", ""),
            url=src.url if hasattr(src, "url") else src.get("url", ""),
            domain=src.domain if hasattr(src, "domain") else src.get("domain", ""),
            tier=src.tier if hasattr(src, "tier") else src.get("tier", ""),
            providers=list(src.providers) if hasattr(src, "providers") else list(src.get("providers", [])),
            tools=list(src.tools) if hasattr(src, "tools") else list(src.get("tools", [])),
        )
        for src in sources
    ]
    return FollowUpRunResponse(
        video_id=video_id,
        summary_id=summary_id,
        status=status,
        answer=answer,
        sources=api_sources,
        meta=meta,
        cache_key=meta.get('cache_key', ''),
        error=error,
    )


def _build_research_sources(sources: list) -> list[ResearchSource]:
    return [
        ResearchSource(
            name=src.name if hasattr(src, "name") else src.get("name", ""),
            url=src.url if hasattr(src, "url") else src.get("url", ""),
            domain=src.domain if hasattr(src, "domain") else src.get("domain", ""),
            tier=src.tier if hasattr(src, "tier") else src.get("tier", ""),
            providers=list(src.providers) if hasattr(src, "providers") else list(src.get("providers", [])),
            tools=list(src.tools) if hasattr(src, "tools") else list(src.get("tools", [])),
        )
        for src in (sources or [])
    ]


def _build_follow_up_thread_turn(turn: Dict[str, Any]) -> FollowUpThreadTurn:
    return FollowUpThreadTurn(
        follow_up_run_id=int(turn.get("run_id") or 0),
        parent_follow_up_run_id=_coerce_optional_int(turn.get("parent_follow_up_run_id")),
        video_id=str(turn.get("video_id") or ""),
        summary_id=_coerce_optional_int(turn.get("summary_id")),
        approved_questions=list(turn.get("approved_questions") or []),
        question_provenance=list(turn.get("question_provenance") or []),
        answer=str(turn.get("answer") or ""),
        sources=_build_research_sources(turn.get("sources") or []),
        status=str(turn.get("status") or "ok"),
        created_at=str(turn.get("created_at") or "") or None,
        meta=dict(turn.get("meta") or {}),
    )


def _resolve_active_follow_up_run(
    *,
    store,
    video_id: str,
    follow_up_run_id: Optional[int] = None,
    summary_id: Optional[int] = None,
    preferred_variant: Optional[str] = None,
    summary: str = "",
    source_context: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], Any]:
    if follow_up_run_id is not None:
        run = store.get_research_run(follow_up_run_id, video_id=video_id)
        return run, None

    resolved = store.resolve_context(
        video_id=video_id,
        summary_id=summary_id,
        summary=summary,
        source_context=dict(source_context or {}),
        preferred_variant=preferred_variant or "deep-research",
    )
    candidate_run_id = _coerce_optional_int((resolved.source_context or {}).get("parent_follow_up_run_id"))
    if candidate_run_id is not None:
        return store.get_research_run(candidate_run_id, video_id=resolved.video_id), resolved
    return None, resolved


@app.get("/api/research/capabilities")
async def research_capabilities(
    _auth: None = Depends(require_auth),
    _service: None = Depends(require_research_service)
):
    """
    Get research service capabilities and configuration.

    Returns information about available providers, models, and settings.
    """
    service = get_research_service()

    try:
        capabilities = service['get_research_capabilities']()
        return JSONResponse(content=capabilities)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capabilities: {str(e)}"
    )


def _coerce_optional_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


@app.post("/api/follow-up-suggestions", response_model=FollowUpSuggestionsResponse)
@app.post("/api/research/follow-up/suggestions", response_model=FollowUpSuggestionsResponse)
async def get_follow_up_suggestions_endpoint(
    request: FollowUpSuggestionsRequest,
    _auth: None = Depends(require_auth),
    _service: None = Depends(require_research_service),
):
    """
    Generate follow-up research suggestions for a summary.

    Uses the planner LLM to analyze the summary and generate contextual
    follow-up questions that users might want to research.

    POST is used instead of GET because the summary can be very large
    and should not be passed in the URL query string.
    """
    service = get_research_service()
    store = get_follow_up_store()

    try:
        resolved = store.resolve_context(
            video_id=request.video_id,
            summary_id=request.summary_id,
            summary=request.summary,
            source_context=dict(request.source_context),
            preferred_variant=request.preferred_variant,
        )
        if resolved.summary_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A persisted summary is required for follow-up suggestions. Provide a valid summary_id or existing video_id."
            )

        # Get suggestions from research service
        raw_suggestions = service['get_follow_up_suggestions'](
            source_context=resolved.source_context,
            summary=resolved.summary,
            max_suggestions=request.max_suggestions,
        )

        store.store_suggestions(
            video_id=request.video_id,
            summary_id=resolved.summary_id,
            suggestions=raw_suggestions,
        )
        store.mark_follow_up_available(resolved.summary_id)

        # Convert to response model
        suggestions = [
            FollowUpSuggestion(
                id=s['id'],
                label=s['label'],
                question=s['question'],
                reason=s['reason'],
                kind=s['kind'],
                priority=s['priority'],
                default_selected=s['default_selected'],
                provenance=s['provenance'],
            )
            for s in raw_suggestions
        ]

        return FollowUpSuggestionsResponse(
            summary_id=resolved.summary_id,
            video_id=request.video_id,
            suggestions=suggestions,
            should_suggest=len(suggestions) > 0,
        )

    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[YTV2 API] Error generating suggestions: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@app.post("/api/follow-up-research", response_model=FollowUpRunResponse)
@app.post("/api/research/follow-up/run", response_model=FollowUpRunResponse)
async def run_follow_up_research_endpoint(
    request: FollowUpRunRequest,
    _auth: None = Depends(require_auth),
    _service: None = Depends(require_research_service),
):
    """
    Run follow-up research based on approved user questions.

    This endpoint:
    1. Takes approved follow-up questions
    2. Consolidates them into a minimal research plan
    3. Executes web research
    4. Returns a sectioned report answering each question

    Results are cached by cache_key for efficient re-use.
    """
    service = get_research_service()
    store = get_follow_up_store()

    logging.info(
        "FOLLOW_UP_RUN request video_id=%s questions=%s",
        request.video_id,
        request.approved_questions,
    )
    print(f"[DEBUG] FOLLOW_UP_RUN request video_id={request.video_id} questions={request.approved_questions}", flush=True)

    try:
        resolved = store.resolve_context(
            video_id=request.video_id,
            summary_id=request.summary_id,
            summary=request.summary,
            source_context=dict(request.source_context),
            preferred_variant=request.preferred_variant,
        )
        if resolved.summary_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A persisted summary is required for follow-up research. Provide a valid summary_id or existing video_id."
            )
        cache_key = service['build_cache_key'](
            video_id=request.video_id,
            summary_id=resolved.summary_id,
            approved_questions=request.approved_questions,
            provider_mode=request.provider_mode,
            depth=request.depth,
        )
        cached = store.get_cached_research(cache_key)
        if cached is not None:
            logging.info(
                "FOLLOW_UP_RUN cache_hit video_id=%s status=%s",
                request.video_id,
                cached.get("status"),
            )
            cached_meta = dict(cached["meta"])
            cached_meta["cache_key"] = cache_key
            cached_meta["cache_hit"] = True
            cached_meta.setdefault("follow_up_run_id", cached["run_id"])
            return _build_follow_up_run_response(
                video_id=request.video_id,
                summary_id=cached["summary_id"],
                status=cached["status"],
                answer=cached["answer"],
                sources=cached["sources"],
                meta=cached_meta,
                error=cached_meta.get("error") if cached["status"] == "error" else None,
            )

        # Run follow-up research
        logging.info(
            "FOLLOW_UP_RUN start video_id=%s summary_id=%s questions=%s",
            request.video_id,
            resolved.summary_id,
            request.approved_questions,
        )
        result = service['run_follow_up_research'](
            source_context=resolved.source_context,
            summary=resolved.summary,
            approved_questions=request.approved_questions,
            question_provenance=request.question_provenance,
            summary_id=resolved.summary_id,
            provider_mode=request.provider_mode,
            tool_mode=request.tool_mode,
            depth=request.depth,
            manual_tools=request.manual_tools,
        )
        parent_follow_up_run_id = _coerce_optional_int(request.parent_follow_up_run_id)
        if parent_follow_up_run_id is None:
            parent_follow_up_run_id = _coerce_optional_int(
                (resolved.source_context or {}).get("parent_follow_up_run_id")
            )
        source_variant = str(
            resolved.variant
            or request.preferred_variant
            or ""
        ).strip().lower()
        if source_variant:
            result.meta["source_summary_variant"] = source_variant
        if resolved.summary_revision is not None:
            result.meta["source_summary_revision"] = resolved.summary_revision
        if parent_follow_up_run_id is not None:
            result.meta["parent_follow_up_run_id"] = parent_follow_up_run_id
        logging.info(
            "FOLLOW_UP_RUN result video_id=%s status=%s cache_hit=%s answer_len=%d",
            request.video_id,
            result.status,
            result.meta.get("cache_hit", False),
            len(result.answer) if result.answer else 0,
        )

        run_id = store.store_research_run(
            video_id=request.video_id,
            summary_id=resolved.summary_id,
            approved_questions=request.approved_questions,
            question_provenance=request.question_provenance,
            result=result,
        )
        result.meta["follow_up_run_id"] = run_id
        result.meta.update(store.create_summary_variant_reference(
            video_id=request.video_id,
            text=result.answer,
        ))
        store.update_research_run_meta(run_id, result.meta)
        store.mark_follow_up_available(resolved.summary_id)

        return _build_follow_up_run_response(
            video_id=request.video_id,
            summary_id=resolved.summary_id,
            status=result.status,
            answer=result.answer,
            sources=result.sources,
            meta=result.meta,
            error=result.meta.get('error') if result.status == 'error' else None,
        )

    except ValueError as e:
        # Validation errors (too many questions, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[YTV2 API] Error running follow-up research: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run follow-up research: {str(e)}"
        )


@app.get("/api/research/follow-up/thread", response_model=FollowUpThreadResponse)
async def get_follow_up_thread_endpoint(
    video_id: str,
    follow_up_run_id: Optional[int] = None,
    summary_id: Optional[int] = None,
    preferred_variant: Optional[str] = None,
    _auth: None = Depends(require_auth),
):
    """Return the persisted deep-research thread for the active run."""
    store = get_follow_up_store()

    try:
        active_run, resolved = _resolve_active_follow_up_run(
            store=store,
            video_id=video_id,
            follow_up_run_id=follow_up_run_id,
            summary_id=summary_id,
            preferred_variant=preferred_variant or "deep-research",
        )
        if active_run is None:
            # No research run yet — load pre-research chat turns if any
            resolved_video_id = str((resolved.video_id if resolved else video_id) or video_id)
            pre_research_turns = store.get_follow_up_chat_turns(video_id=resolved_video_id)
            chat_turns = [
                FollowUpChatTurnResponse(**turn) for turn in pre_research_turns
            ]
            return FollowUpThreadResponse(
                video_id=resolved_video_id,
                root_follow_up_run_id=None,
                current_follow_up_run_id=None,
                turns=[],
                chat_turns=chat_turns,
            )

        resolved_video_id = str((active_run.get("video_id") or (resolved.video_id if resolved else video_id)) or video_id)
        thread = store.get_research_thread(int(active_run["run_id"]), video_id=resolved_video_id)
        turns = [_build_follow_up_thread_turn(turn) for turn in thread]
        root_run_id = turns[0].follow_up_run_id if turns else None
        current_run_id = turns[-1].follow_up_run_id if turns else None

        # Load persisted chat turns for the current run
        chat_turns_raw = store.get_follow_up_chat_turns(int(active_run["run_id"]), video_id=resolved_video_id)
        chat_turns = [
            FollowUpChatTurnResponse(**turn) for turn in chat_turns_raw
        ]

        return FollowUpThreadResponse(
            video_id=resolved_video_id,
            root_follow_up_run_id=root_run_id,
            current_follow_up_run_id=current_run_id,
            turns=turns,
            chat_turns=chat_turns,
        )
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load Deep Research thread: {str(e)}"
        )


@app.post("/api/research/follow-up/chat", response_model=FollowUpChatResponse)
async def answer_follow_up_chat_endpoint(
    request: FollowUpChatRequest,
    _auth: None = Depends(require_auth),
    _service: None = Depends(require_research_service),
):
    """Answer a lightweight question using the current deep-research report or summary."""
    service = get_research_service()
    store = get_follow_up_store()

    try:
        active_run, resolved = _resolve_active_follow_up_run(
            store=store,
            video_id=request.video_id,
            follow_up_run_id=request.follow_up_run_id,
            summary_id=request.summary_id,
            preferred_variant=request.preferred_variant or "deep-research",
            summary=request.summary,
            source_context=dict(request.source_context),
        )

        # Determine the report context: research run if available, else resolved summary
        if active_run is not None:
            report_answer = str(active_run.get("answer") or request.summary or "")
            report_sources = list(active_run.get("sources") or [])
            report_video_id = str(active_run.get("video_id") or request.video_id)
            report_run_id = int(active_run["run_id"])
        elif resolved is not None:
            report_answer = str(resolved.summary or request.summary or "")
            report_sources = []
            report_video_id = str(resolved.video_id or request.video_id)
            report_run_id = None
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No summary or research context found for this item."
            )

        resolved_source_context = dict(request.source_context or {})
        if resolved is not None:
            resolved_source_context = dict(resolved.source_context or {})
        resolved_source_context.setdefault("video_id", report_video_id)
        resolved_source_context.setdefault("id", report_video_id)

        thread_turns = []
        if report_run_id is not None:
            thread_turns = store.get_research_thread(report_run_id, video_id=report_video_id)

        answer, llm_info, serialized_sources = service["answer_follow_up_chat"](
            source_context=resolved_source_context,
            report_answer=report_answer,
            report_sources=report_sources,
            user_question=request.question,
            history=[{"role": turn.role, "content": turn.content} for turn in request.history],
            thread_turns=thread_turns,
        )
        response_meta = {
            "mode": "report-chat",
            "follow_up_run_id": report_run_id,
            "source_count": len(serialized_sources),
            "thread_turn_count": len(thread_turns),
            "llm_provider": llm_info.get("llm_provider", "unknown"),
            "llm_model": llm_info.get("llm_model", "unknown"),
        }

        persisted = False
        if request.persist:
            try:
                resolved_summary_id = resolved.summary_id if resolved else None
                store.store_follow_up_chat_turn(
                    follow_up_run_id=report_run_id,
                    video_id=report_video_id,
                    question=request.question,
                    answer=answer,
                    sources=[s if isinstance(s, dict) else s.__dict__ for s in serialized_sources],
                    chat_meta=response_meta,
                    summary_id=resolved_summary_id,
                )
                persisted = True
            except Exception as persist_err:
                logger.warning("Failed to persist chat turn for run %s: %s", report_run_id, persist_err)

        return FollowUpChatResponse(
            video_id=report_video_id,
            follow_up_run_id=report_run_id,
            answer=answer,
            sources=_build_research_sources(serialized_sources),
            meta=response_meta,
            persisted=persisted,
        )
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer report chat question: {str(e)}"
        )


@app.get("/api/research/follow-up/cached", response_model=FollowUpCachedResponse)
async def check_cached_research(
    cache_key: str,
    _auth: None = Depends(require_auth),
):
    """
    Check if follow-up research results are cached.

    Queries follow_up_research_runs for a prior matching run.
    """
    store = get_follow_up_store()
    cached = store.get_cached_research(cache_key)
    if cached is None:
        return FollowUpCachedResponse(cached=False, result=None, cache_key=cache_key)

    cached_meta = dict(cached["meta"])
    cached_meta["cache_key"] = cache_key
    cached_meta["cache_hit"] = True
    cached_meta.setdefault("follow_up_run_id", cached["run_id"])
    return FollowUpCachedResponse(
        cached=True,
        result=_build_follow_up_run_response(
            video_id=cached["video_id"],
            summary_id=cached["summary_id"],
            status=cached["status"],
            answer=cached["answer"],
            sources=cached["sources"],
            meta=cached_meta,
            error=cached_meta.get("error") if cached["status"] == "error" else None,
        ),
        cache_key=cache_key,
    )


@app.post("/api/research/follow-up/chat/stream")
async def stream_follow_up_chat_endpoint(
    request: FollowUpChatRequest,
    _auth: None = Depends(require_auth),
    _service: None = Depends(require_research_service),
):
    """Stream a grounded chat answer using Mercury 2 diffusing."""
    service = get_research_service()
    store = get_follow_up_store()

    active_run, resolved = _resolve_active_follow_up_run(
        store=store,
        video_id=request.video_id,
        follow_up_run_id=request.follow_up_run_id,
        summary_id=request.summary_id,
        preferred_variant=request.preferred_variant or "deep-research",
        summary=request.summary,
        source_context=dict(request.source_context),
    )

    if active_run is not None:
        report_answer = str(active_run.get("answer") or request.summary or "")
        report_sources = list(active_run.get("sources") or [])
        report_video_id = str(active_run.get("video_id") or request.video_id)
        report_run_id = int(active_run["run_id"])
    elif resolved is not None:
        report_answer = str(resolved.summary or request.summary or "")
        report_sources = []
        report_video_id = str(resolved.video_id or request.video_id)
        report_run_id = None
    else:
        raise HTTPException(status_code=404, detail="No summary or research context found for this item.")

    resolved_source_context = dict(request.source_context or {})
    if resolved is not None:
        resolved_source_context = dict(resolved.source_context or {})
    resolved_source_context.setdefault("video_id", report_video_id)
    resolved_source_context.setdefault("id", report_video_id)

    thread_turns = []
    if report_run_id is not None:
        thread_turns = store.get_research_thread(report_run_id, video_id=report_video_id)

    full_answer = []

    def event_stream():
        for event_type, data in service["stream_report_chat"](
            source_context=resolved_source_context,
            report_answer=report_answer,
            report_sources=report_sources,
            user_question=request.question,
            history=[{"role": turn.role, "content": turn.content} for turn in request.history],
            thread_turns=thread_turns,
        ):
            if event_type == "token":
                full_answer.append(data)
                yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
            elif event_type == "reasoning":
                yield f"data: {json.dumps({'type': 'reasoning', 'content': data})}\n\n"
            elif event_type == "done":
                # Persist after streaming completes
                answer_text = "".join(full_answer)
                if request.persist and answer_text:
                    try:
                        store.store_follow_up_chat_turn(
                            follow_up_run_id=report_run_id,
                            video_id=report_video_id,
                            question=request.question,
                            answer=answer_text,
                            sources=[],
                            chat_meta={"mode": "report-chat-stream", "follow_up_run_id": report_run_id},
                            summary_id=resolved.summary_id if resolved else None,
                        )
                    except Exception as persist_err:
                        logger.warning("Failed to persist streamed chat turn: %s", persist_err)
                yield f"data: {json.dumps({'type': 'done', 'persisted': request.persist})}\n\n"
            elif event_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': data})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/api/research/follow-up/chat-turns/{turn_id}")
async def delete_follow_up_chat_turn_endpoint(
    turn_id: int,
    video_id: Optional[str] = None,
    _auth: None = Depends(require_auth),
):
    """Delete a single persisted chat turn."""
    store = get_follow_up_store()
    deleted = store.delete_follow_up_chat_turn(turn_id, video_id=video_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat turn {turn_id} not found."
        )
    return {"deleted": True, "turn_id": turn_id}


@app.delete("/api/research/follow-up/chat-turns")
async def delete_follow_up_chat_turns_by_run_endpoint(
    follow_up_run_id: Optional[int] = None,
    video_id: Optional[str] = None,
    _auth: None = Depends(require_auth),
):
    """Delete all chat turns for a research run or pre-research video."""
    store = get_follow_up_store()
    if follow_up_run_id is not None:
        deleted_count = store.delete_follow_up_chat_turns_by_run(follow_up_run_id)
        return {"deleted": True, "follow_up_run_id": follow_up_run_id, "count": deleted_count}
    elif video_id:
        deleted_count = store.delete_follow_up_chat_turns_by_video(video_id)
        return {"deleted": True, "video_id": video_id, "count": deleted_count}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either follow_up_run_id or video_id."
        )


@app.get("/")
async def root():
    """Root endpoint - service info."""
    return {
        "service": "YTV2 Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/process",
            "transcript": "/api/transcript",
            "health": "/health",
            "research": {
                "capabilities": "/api/research/capabilities",
                "legacy_follow_up_suggestions": "/api/follow-up-suggestions (POST)",
                "legacy_follow_up_research": "/api/follow-up-research (POST)",
                "follow_up_suggestions": "/api/research/follow-up/suggestions (POST)",
                "run_follow_up": "/api/research/follow-up/run (POST)",
                "check_cached": "/api/research/follow-up/cached"
            }
        }
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid request format", "details": str(exc), "status": "error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    import traceback
    print(f"[YTV2 API] Unexpected error: {exc}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "status": "error"}
    )


# ---- Audio On-Demand Generation ----

@app.post("/api/audio/generate")
async def generate_audio_artifact(
    video_id: str = Body(..., embed=True),
    mode: str = Body(..., embed=True),
    scope: str = Body("summary_active"),
    variant_slug: Optional[str] = Body(None),
    authorization: Optional[str] = Header(None),
):
    """Generate an audio artifact for a video+mode+scope.

    Pipeline: resolve source text → LLM narrative transform → TTS → MP3 file → artifact row.
    """
    await require_auth(authorization)

    from .audio_store import AudioStore, compute_source_hash, build_tts_config_tag
    from research_api.research_service.tts_provider import get_tts_provider
    from research_api.research_service.llm import chat_text

    store = AudioStore()

    # 1. Resolve source text
    source_text = store.resolve_source_text(video_id, scope, variant_slug, mode=mode)
    if not source_text:
        raise HTTPException(status_code=400, detail="No source text found for the given scope")

    source_label = store.resolve_source_label(video_id, scope, variant_slug, mode=mode)

    # 2. Compute source hash and check cache
    tts_config_tag = build_tts_config_tag()
    source_hash = compute_source_hash(mode, scope, source_text, tts_config=tts_config_tag)
    cached = store.get_artifact(video_id, mode, scope)
    if cached and cached.get("status") == "ready" and cached.get("source_hash") == source_hash:
        return {
            "status": "ready",
            "cached": True,
            "audio_url": cached["audio_url"],
            "duration_seconds": cached.get("duration_seconds"),
            "source_label": cached.get("source_label"),
        }

    # 3. Upsert artifact as queued
    store.upsert_artifact(video_id, mode, scope, source_hash,
                          status="queued", source_label=source_label)

    try:
        # 4. LLM narrative transform (only for text-heavy modes)
        llm_effort = os.getenv("AUDIO_LLM_REASONING_EFFORT", "low")
        if mode == "audio_current" and len(source_text) > 200:
            narrative_prompt = (
                "Convert this summary into a natural, text-to-speech friendly narration.\n\n"
                f"{source_text[:4000]}\n\n"
                "Rules:\n"
                "- Use flowing paragraphs only (no bullets/headings/code fences/emojis).\n"
                "- Smooth transitions between topics; keep proper nouns/numbers accurate.\n"
                "- Maintain all key information and conclusions.\n"
                "- End with a single sentence beginning \"Bottom line: ...\".\n"
                "- Length target: 180-350 words.\n"
            )
            narrative, _, _ = chat_text(
                messages=[{"role": "user", "content": narrative_prompt}],
                max_tokens=1024,
                reasoning_effort=llm_effort,
                temperature=0.5,
                timeout=30,
            )
            tts_text = narrative if narrative else source_text
        elif mode == "audio_briefing":
            # Briefings get a more structured transform
            narrative_prompt = (
                "Synthesize this content into a cohesive audio briefing.\n\n"
                f"{source_text[:6000]}\n\n"
                "Rules:\n"
                "- Write as a news briefing narrator would speak it.\n"
                "- Flowing paragraphs, no bullets or headings.\n"
                "- Cover key findings and conclusions.\n"
                "- Length target: 300-600 words.\n"
            )
            narrative, _, _ = chat_text(
                messages=[{"role": "user", "content": narrative_prompt}],
                max_tokens=2048,
                reasoning_effort=llm_effort,
                temperature=0.5,
                timeout=45,
            )
            tts_text = narrative if narrative else source_text
        else:
            tts_text = source_text

        if not tts_text:
            raise RuntimeError("LLM transform returned empty text")

        # 5. Update status to generating
        store.update_status(video_id, mode, scope, "generating")

        # 6. TTS generation
        tts = get_tts_provider()
        audio_bytes = tts.generate(tts_text, voice="fable", output_format="mp3")
        provider_name = tts.name()

        # 7. Write file to data/exports/audio/ so dashboard can serve it
        #    via its existing ../backend/data:/app/backend-data volume mount.
        from pathlib import Path
        export_dir = Path("data/exports/audio")
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = f"audio_{video_id}_{mode}_{source_hash[:8]}.mp3"
        filepath = export_dir / filename
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        audio_url = f"/exports/audio/{filename}"

        # 8. Estimate duration (tts-1 ~1MB per minute for MP3)
        duration_seconds = int(len(audio_bytes) / 16000)

        # 9. Update artifact to ready
        store.update_status(video_id, mode, scope, "ready",
                            audio_url=audio_url,
                            duration_seconds=duration_seconds,
                            provider=provider_name,
                            source_label=source_label)

        return {
            "status": "ready",
            "cached": False,
            "audio_url": audio_url,
            "duration_seconds": duration_seconds,
            "source_label": source_label,
        }

    except Exception as e:
        logger.error(f"Audio generation failed for {video_id}/{mode}/{scope}: {e}")
        store.update_status(video_id, mode, scope, "failed",
                            error_message=str(e)[:500])
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)},
        )


@app.post("/api/audio/generate/stream")
async def generate_audio_artifact_stream(
    video_id: str = Body(..., embed=True),
    mode: str = Body(..., embed=True),
    scope: str = Body("summary_active"),
    variant_slug: Optional[str] = Body(None),
    authorization: Optional[str] = Header(None),
):
    """Stream audio generation: sends chunks to client while storing the full file.

    Uses Fish Audio streaming when available, falls back to buffered generation.
    Returns audio/mp3 stream with the full file saved for subsequent cached plays.
    """
    await require_auth(authorization)

    from .audio_store import AudioStore, compute_source_hash, build_tts_config_tag
    from research_api.research_service.tts_provider import get_tts_provider
    from research_api.research_service.llm import chat_text

    store = AudioStore()

    # 1. Resolve source text
    source_text = store.resolve_source_text(video_id, scope, variant_slug, mode=mode)
    if not source_text:
        raise HTTPException(status_code=400, detail="No source text found for the given scope")

    source_label = store.resolve_source_label(video_id, scope, variant_slug, mode=mode)

    # 2. Check cache — if ready, redirect to stored file
    source_hash = compute_source_hash(mode, scope, source_text)
    cached = store.get_artifact(video_id, mode, scope)
    if cached and cached.get("status") == "ready" and cached.get("source_hash") == source_hash:
        # Already have this artifact cached — return the URL so frontend fetches the file
        return JSONResponse(content={
            "status": "ready",
            "cached": True,
            "audio_url": cached["audio_url"],
            "duration_seconds": cached.get("duration_seconds"),
            "source_label": cached.get("source_label"),
        })

    # 3. Upsert artifact as queued
    store.upsert_artifact(video_id, mode, scope, source_hash,
                          status="queued", source_label=source_label)

    try:
        # 4. LLM narrative transform
        llm_effort = os.getenv("AUDIO_LLM_REASONING_EFFORT", "low")
        if mode == "audio_current" and len(source_text) > 200:
            narrative_prompt = (
                "Convert this summary into a natural, text-to-speech friendly narration.\n\n"
                f"{source_text[:4000]}\n\n"
                "Rules:\n"
                "- Use flowing paragraphs only (no bullets/headings/code fences/emojis).\n"
                "- Smooth transitions between topics; keep proper nouns/numbers accurate.\n"
                "- Maintain all key information and conclusions.\n"
                "- End with a single sentence beginning \"Bottom line: ...\".\n"
                "- Length target: 180-350 words.\n"
            )
            narrative, _, _ = chat_text(
                messages=[{"role": "user", "content": narrative_prompt}],
                max_tokens=1024,
                reasoning_effort=llm_effort,
                temperature=0.5,
                timeout=30,
            )
            tts_text = narrative if narrative else source_text
        elif mode == "audio_briefing":
            narrative_prompt = (
                "Synthesize this content into a cohesive audio briefing.\n\n"
                f"{source_text[:6000]}\n\n"
                "Rules:\n"
                "- Write as a news briefing narrator would speak it.\n"
                "- Flowing paragraphs, no bullets or headings.\n"
                "- Cover key findings and conclusions.\n"
                "- Length target: 300-600 words.\n"
            )
            narrative, _, _ = chat_text(
                messages=[{"role": "user", "content": narrative_prompt}],
                max_tokens=2048,
                reasoning_effort=llm_effort,
                temperature=0.5,
                timeout=45,
            )
            tts_text = narrative if narrative else source_text
        else:
            tts_text = source_text

        if not tts_text:
            raise RuntimeError("LLM transform returned empty text")

        store.update_status(video_id, mode, scope, "generating")

        # 5. TTS generation
        tts = get_tts_provider()
        provider_name = tts.name()
        from pathlib import Path
        export_dir = Path("data/exports/audio")
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = f"audio_{video_id}_{mode}_{source_hash[:8]}.mp3"
        filepath = export_dir / filename
        audio_url = f"/exports/audio/{filename}"

        if tts.supports_streaming():
            # Stream from provider, collect bytes, save file when done
            def event_stream():
                collected = bytearray()
                try:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'generating'})}\n\n"

                    for chunk in tts.generate_stream(tts_text, voice="fable", output_format="mp3"):
                        collected.extend(chunk)

                    # Save the full file
                    with open(filepath, "wb") as f:
                        f.write(collected)

                    duration_seconds = int(len(collected) / 16000)
                    store.update_status(video_id, mode, scope, "ready",
                                        audio_url=audio_url,
                                        duration_seconds=duration_seconds,
                                        provider=provider_name,
                                        source_label=source_label)

                    yield f"data: {json.dumps({'type': 'done', 'audio_url': audio_url, 'duration_seconds': duration_seconds, 'source_label': source_label})}\n\n"

                except Exception as e:
                    logger.error(f"Audio streaming failed for {video_id}/{mode}/{scope}: {e}")
                    store.update_status(video_id, mode, scope, "failed",
                                        error_message=str(e)[:500])
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        else:
            # Non-streaming provider: generate, save, return URL
            audio_bytes = tts.generate(tts_text, voice="fable", output_format="mp3")
            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            duration_seconds = int(len(audio_bytes) / 16000)
            store.update_status(video_id, mode, scope, "ready",
                                audio_url=audio_url,
                                duration_seconds=duration_seconds,
                                provider=provider_name,
                                source_label=source_label)

            return JSONResponse(content={
                "status": "ready",
                "cached": False,
                "audio_url": audio_url,
                "duration_seconds": duration_seconds,
                "source_label": source_label,
            })

    except Exception as e:
        logger.error(f"Audio generation failed for {video_id}/{mode}/{scope}: {e}")
        store.update_status(video_id, mode, scope, "failed",
                            error_message=str(e)[:500])
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)},
        )


def main():
    """Run the API server directly."""
    import uvicorn
    port = int(os.getenv('YTV2_API_PORT', '6453'))
    host = os.getenv('YTV2_API_HOST', '127.0.0.1')
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

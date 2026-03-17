"""
YTV2 HTTP API - Multi-channel gateway endpoints.

This FastAPI application provides internal HTTP endpoints for processing
YouTube videos, Reddit threads, and web pages. It's designed to be called
by multi-channel gateways like Clawdbot.

Security: Binds to 127.0.0.1 only - accessible only from within the NAS.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Header, Body, Depends
from fastapi.responses import JSONResponse
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
            )
            from research_service.follow_up import build_cache_key
            _research_service = {
                'get_follow_up_suggestions': get_follow_up_suggestions,
                'run_follow_up_research': run_follow_up_research,
                'get_research_capabilities': get_research_capabilities,
                'build_cache_key': build_cache_key,
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


def main():
    """Run the API server directly."""
    import uvicorn
    port = int(os.getenv('YTV2_API_PORT', '6453'))
    host = os.getenv('YTV2_API_HOST', '127.0.0.1')
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

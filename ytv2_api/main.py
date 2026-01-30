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

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Header
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .models import ProcessRequest, ProcessResponse, HealthResponse

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


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="ytv2-api",
        version="1.0.0"
    )


@app.get("/")
async def root():
    """Root endpoint - service info."""
    return {
        "service": "YTV2 Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/process",
            "health": "/health"
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

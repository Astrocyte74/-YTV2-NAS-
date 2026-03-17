"""
Pydantic models for YTV2 API requests and responses.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List, Literal


# Research configuration constants
class ProviderMode:
    AUTO = "auto"
    BRAVE = "brave"
    TAVILY = "tavily"
    BOTH = "both"


class ToolMode:
    AUTO = "auto"
    SAFE = "safe"
    DEEP = "deep"
    MANUAL = "manual"


class ResearchDepth:
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"


class ProcessRequest(BaseModel):
    """Request model for URL processing."""

    url: str = Field(..., description="URL to process (YouTube, Reddit, or web page)")
    summary_type: str = Field(default="comprehensive", description="Type of summary to generate")
    user_id: str = Field(default="clawdbot", description="User or service identifier")
    channel: str = Field(default="unknown", description="Source channel (telegram, whatsapp, discord, etc.)")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "summary_type": "comprehensive",
                    "user_id": "user123",
                    "channel": "whatsapp"
                }
            ]
        }


class ProcessResponse(BaseModel):
    """Response model for processed content."""

    content_id: str = Field(..., description="Unique content identifier")
    status: str = Field(..., description="Processing status (completed, failed, pending)")
    summary: Optional[str] = Field(None, description="Generated summary text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")
    dashboard_url: Optional[str] = Field(None, description="Link to dashboard if available")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    source_type: str = Field(..., description="Type of content (youtube, reddit, web)")


class TranscriptResponse(BaseModel):
    """Response model for transcript-only requests."""

    content_id: str = Field(..., description="Unique content identifier (video ID, etc.)")
    status: str = Field(..., description="Processing status (completed, failed)")
    transcript: Optional[str] = Field(None, description="Full transcript text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata (title, channel, duration, etc.)")
    chapters: Optional[list] = Field(None, description="Video chapters with timestamps")
    segments: Optional[list] = Field(None, description="Transcript segments with timestamps")
    language: Optional[str] = Field(None, description="Transcript language code")
    duration: Optional[int] = Field(None, description="Video duration in seconds")
    error: Optional[str] = Field(None, description="Error message if extraction failed")
    source_type: str = Field(..., description="Type of content (youtube, reddit, web)")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")


# =============================================================================
# Follow-Up Research Models
# =============================================================================

class FollowUpSuggestion(BaseModel):
    """A suggested follow-up research question."""

    id: str = Field(..., description="Unique suggestion identifier")
    label: str = Field(..., description="User-facing label for the suggestion")
    question: str = Field(..., description="The actual research question")
    reason: str = Field(..., description="Why this suggestion is relevant")
    kind: Literal["current_state", "pricing", "comparison", "alternatives", "fact_check", "background", "what_changed"] = Field(
        ..., description="Question category"
    )
    priority: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    default_selected: bool = Field(..., description="Whether this should be pre-selected")
    provenance: Literal["suggested", "preset", "custom"] = Field(..., description="Where the suggestion came from")


class FollowUpSuggestionsRequest(BaseModel):
    """Request model for generating follow-up suggestions."""

    video_id: str = Field(..., description="Content identifier")
    summary: str = Field(..., description="The existing summary text")
    summary_id: Optional[int] = Field(None, description="Summary ID if available")
    preferred_variant: Optional[str] = Field(None, description="Preferred persisted summary variant to resolve")
    source_context: Dict[str, Any] = Field(default_factory=dict, description="Additional source context")
    max_suggestions: int = Field(default=3, ge=1, le=5, description="Maximum number of suggestions to generate")


class FollowUpSuggestionsResponse(BaseModel):
    """Response model for follow-up suggestions."""

    summary_id: Optional[int] = Field(None, description="Summary ID if available")
    video_id: str = Field(..., description="Content identifier")
    suggestions: List[FollowUpSuggestion] = Field(default_factory=list, description="List of suggested follow-up questions")
    should_suggest: bool = Field(default=True, description="Whether follow-up research is recommended")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class FollowUpRunRequest(BaseModel):
    """Request model for running follow-up research."""

    video_id: str = Field(..., description="Content identifier (video ID, URL hash, etc.)")
    summary_id: Optional[int] = Field(None, description="Specific summary revision ID")
    preferred_variant: Optional[str] = Field(None, description="Preferred persisted summary variant to resolve")
    summary: str = Field(..., description="The existing summary text")
    source_context: Dict[str, Any] = Field(default_factory=dict, description="Information about original content")
    approved_questions: List[str] = Field(..., min_length=1, max_length=3, description="User-approved research questions")
    question_provenance: Optional[List[str]] = Field(None, description="Where each question came from")
    provider_mode: Literal["auto", "brave", "tavily", "both"] = Field(default="auto", description="Research provider mode")
    tool_mode: Literal["auto", "safe", "deep", "manual"] = Field(default="auto", description="Research tool mode")
    depth: Literal["quick", "balanced", "deep"] = Field(default="balanced", description="Research depth")
    manual_tools: Optional[Dict[str, List[str]]] = Field(None, description="Manual tool selection per provider")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "video_id": "abc123",
                    "summary_id": 456,
                    "preferred_variant": "key-insights",
                    "summary": "This video reviews Cursor AI, covering its pricing and features...",
                    "source_context": {
                        "title": "Cursor AI Review",
                        "url": "https://youtube.com/watch?v=abc123",
                        "published_at": "2024-06-01T00:00:00Z",
                        "type": "youtube"
                    },
                    "approved_questions": ["What is the current pricing?"],
                    "question_provenance": ["suggested"],
                    "provider_mode": "auto",
                    "depth": "balanced"
                }
            ]
        }


class ResearchSource(BaseModel):
    """A research source (website, paper, etc.)."""

    name: str = Field(..., description="Source name/title")
    url: str = Field(..., description="Source URL")
    domain: str = Field(..., description="Source domain")
    tier: str = Field(..., description="Source quality tier")
    providers: List[str] = Field(default_factory=list, description="Providers that found this source")
    tools: List[str] = Field(default_factory=list, description="Tools used to find this source")


class FollowUpRunResponse(BaseModel):
    """Response model for follow-up research run."""

    video_id: str = Field(..., description="Content identifier")
    summary_id: Optional[int] = Field(None, description="Summary revision ID")
    status: str = Field(..., description="Research status (ok, fallback, error)")
    answer: str = Field(..., description="The follow-up research report")
    sources: List[ResearchSource] = Field(default_factory=list, description="Research sources")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Research metadata")
    cache_key: str = Field(..., description="Cache key for this research run")
    error: Optional[str] = Field(None, description="Error message if research failed")


class FollowUpCachedResponse(BaseModel):
    """Response model for cached follow-up research lookup."""

    cached: bool = Field(..., description="Whether a cached result was found")
    result: Optional[FollowUpRunResponse] = Field(None, description="Cached result if found")
    cache_key: str = Field(..., description="Cache key used for lookup")

"""
Pydantic models for YTV2 API requests and responses.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any


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

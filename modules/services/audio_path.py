from __future__ import annotations

from typing import Dict, Optional


def build_audio_path(content: Dict) -> Optional[str]:
    """
    Return the canonical, root‑relative audio path for a piece of content.

    Rules:
    - YouTube/Web (any item with a stable video_id): /exports/by_video/<videoId>.mp3
    - Reddit legacy: /exports/audio/reddit<file_stem>.mp3

    Expects a dict that may contain either universal fields or legacy
    shapes; it safely handles missing keys and returns None when it
    cannot derive a stable path.
    """
    if not isinstance(content, dict):
        return None

    # Prefer explicit video_id
    video_id = (content.get("video_id") or "").strip()
    content_id = (content.get("id") or "").strip()

    # Detect reddit legacy via a conventional file_stem
    # file_stem is used for legacy reddit paths and is already sanitized upstream
    file_stem = (content.get("file_stem") or "").strip()

    # If reddit legacy is indicated by file_stem convention
    # Keep existing convention: /exports/audio/reddit<file_stem>.mp3
    if file_stem and (content.get("content_source") == "reddit" or content_id.startswith("reddit:") or "reddit" in file_stem.lower()):
        return f"/exports/audio/reddit{file_stem}.mp3"

    # Fallback to by_video for any stable video_id (YouTube/Web)
    if video_id:
        return f"/exports/by_video/{video_id}.mp3"

    # Attempt to derive video_id from id formats like yt:<id>, web:<slug>
    if content_id and ":" in content_id:
        _, tail = content_id.split(":", 1)
        if tail:
            return f"/exports/by_video/{tail}.mp3"

    return None


__all__ = ["build_audio_path"]


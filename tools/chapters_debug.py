#!/usr/bin/env python3
"""
Quick helper to inspect YouTube chapter metadata and align transcript segments.

Usage:
    python tools/chapters_debug.py --url https://www.youtube.com/watch?v=...
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from youtube_summarizer import YouTubeSummarizer

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None  # type: ignore[assignment]


def load_transcript_segments(video_id: str, languages: List[str]) -> List[Dict[str, float]]:
    if YouTubeTranscriptApi is None:
        return []
    api = YouTubeTranscriptApi()
    for lang in languages:
        try:
            return api.fetch(video_id, [lang])
        except Exception:
            continue
    try:
        transcript_list = api.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(languages)
        return transcript.fetch()
    except Exception:
        return []


def aggregate_text(segments: List[Dict[str, float]], start: float, end: float) -> str:
    if not segments:
        return ""
    parts: List[str] = []
    for entry in segments:
        if hasattr(entry, "start"):
            begin = float(getattr(entry, "start", 0.0))
            text = str(getattr(entry, "text", "")).strip()
        else:
            begin = float(entry.get("start", 0.0))
            text = str(entry.get("text", "")).strip()
        if begin < start:
            continue
        if begin >= end:
            break
        if text:
            parts.append(text.replace("\n", " "))
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect chapter metadata for a YouTube video.")
    parser.add_argument("--url", required=True, help="Full YouTube watch URL")
    args = parser.parse_args()

    summarizer = YouTubeSummarizer()
    info = summarizer.extract_transcript(args.url)
    metadata: Dict[str, object] = info.get("metadata") or {}
    chapters = metadata.get("chapters") or []
    if not isinstance(chapters, list) or not chapters:
        print("No chapters available for this video.")
        return 0

    video_id = metadata.get("video_id") or summarizer._extract_video_id(args.url)
    language = info.get("transcript_language") or "en"
    segments = load_transcript_segments(str(video_id), [str(language)])

    print(f"Found {len(chapters)} chapters for video {video_id}:")
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = chapter.get("title") or "Untitled chapter"
        start = float(chapter.get("start_time", 0.0))
        end = float(chapter.get("end_time", start))
        text_preview = aggregate_text(segments, start, end)
        preview = text_preview[:160] + "…" if len(text_preview) > 160 else text_preview
        print(f"- {title} [{start:>8.1f} → {end:>8.1f}]")
        if preview:
            print(f"  {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Reformat and re-upload summary HTML for an existing report into Postgres.

- Finds the report JSON under data/reports by video_id (substring match)
- Uses PostgresWriter to upsert content + summaries, relying on the updated
  format_summary_html to produce improved HTML (headings, bullets, Bottom line)

Usage:
  python3 tools/reupload_report_html.py --video-id web:0560bdd21062f9c7e76e8efd
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.postgres_writer import PostgresWriter


def load_report_by_video_id(video_id: str) -> Dict[str, Any]:
    root = Path("data/reports")
    if not root.is_dir():
        raise FileNotFoundError("data/reports directory not found")
    # Scan for a file containing the video_id
    candidates = [p for p in root.glob("*.json") if video_id in p.name]
    if not candidates:
        # Fallback to open and check content
        for p in root.glob("*.json"):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if (raw.get("id") or "") == video_id or (raw.get("metadata", {}).get("video_id") or "") == video_id:
                    candidates.append(p)
                    break
            except Exception:
                continue
    if not candidates:
        raise FileNotFoundError(
            f"No report JSON found for video_id: {video_id}. Checked under {root}"
        )
    # Pick the most recent by mtime
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def to_content_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    # Build a minimal content_data payload that PostgresWriter understands
    video_id = report.get("id") or report.get("metadata", {}).get("video_id")
    meta = report.get("metadata", {})
    title = report.get("title") or meta.get("title") or "Untitled"
    channel_name = meta.get("channel") or meta.get("uploader")
    canonical_url = report.get("canonical_url") or report.get("url")
    thumbnail_url = report.get("thumbnail_url") or meta.get("thumbnail")
    duration_seconds = report.get("duration_seconds") or meta.get("duration") or 0
    language = report.get("summary_language") or report.get("original_language") or meta.get("language")

    summary = report.get("summary") or {}
    # Keep only active summary text/type; PostgresWriter will format HTML
    summary_section = {
        "summary": summary.get("summary") or "",
        "summary_type": summary.get("summary_type") or "comprehensive",
        "language": language,
    }

    return {
        "video_id": video_id,
        "id": video_id,
        "title": title,
        "channel_name": channel_name,
        "canonical_url": canonical_url,
        "thumbnail_url": thumbnail_url,
        "duration_seconds": duration_seconds,
        "language": language,
        "summary": summary_section,
        # optionally pass analysis for completeness
        "analysis_json": report.get("analysis") or None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reformat and re-upload summary HTML for one report")
    ap.add_argument("--video-id", required=True, help="Video id, e.g. web:0560bdd21062f9c7e76e8efd")
    args = ap.parse_args()

    # Ensure DB env present
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL not set; cannot write to Postgres")
        return 2

    report = load_report_by_video_id(args.video_id)
    content = to_content_payload(report)
    w = PostgresWriter()
    payload = w._to_db_payload(content)
    # Show what HTML will be written for sanity
    sv = payload.get("summary_variants") or []
    if sv:
        print("Variant:", sv[0].get("variant"), "Chars:", len(sv[0].get("html") or ""))
        print("HTML preview:\n", (sv[0].get("html") or "")[:600])
    res = w.upload_content(content)
    print("Upsert:", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Scan recent reports and enqueue summary image jobs for any without an image.

Usage:
  python tools/enqueue_missing_images.py --limit 25
  python tools/enqueue_missing_images.py --since-hours 24
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "data" / "reports"


def load_report(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def has_image(report: Dict) -> bool:
    s = report.get("summary") or {}
    if isinstance(s, dict):
        if s.get("summary_image") or s.get("summary_image_url"):
            return True
    # Also check top-level convenience fields sometimes used
    if report.get("summary_image") or report.get("summary_image_url"):
        return True
    return False


def extract_content_payload(report: Dict) -> Dict:
    # Use the same shape youtube_summarizer returns when passing to maybe_generate_summary_image
    content: Dict = {
        "id": report.get("id") or (report.get("video") or {}).get("video_id") or "",
        "title": report.get("title") or (report.get("metadata") or {}).get("title") or (report.get("video") or {}).get("title") or "",
        "metadata": report.get("metadata") or report.get("video") or {},
        "summary": report.get("summary") or {},
        "analysis": report.get("analysis") or {},
    }
    # Normalize id
    if not content.get("id"):
        vid = (report.get("video") or {}).get("video_id") or (report.get("video") or {}).get("id")
        if vid:
            content["id"] = vid
    return content


def main() -> int:
    ap = argparse.ArgumentParser(description="Enqueue image jobs for reports missing images")
    ap.add_argument("--limit", type=int, default=20, help="Max reports to enqueue (most recent first)")
    ap.add_argument("--since-hours", type=int, default=None, help="Only consider reports modified within the last N hours")
    args = ap.parse_args()

    if not REPORTS_DIR.exists():
        print(f"No reports dir: {REPORTS_DIR}")
        return 0

    # Collect report files sorted by mtime desc
    items: List[Path] = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Filter by recency if requested
    if args.since_hours is not None:
        import time
        cutoff = time.time() - (args.since_hours * 3600)
        items = [p for p in items if p.stat().st_mtime >= cutoff]

    enqueued = 0
    skipped_have = 0
    scanned = 0

    from modules import image_queue

    for path in items:
        if args.limit is not None and enqueued >= args.limit:
            break
        report = load_report(path)
        if not report:
            continue
        scanned += 1
        if has_image(report):
            skipped_have += 1
            continue
        payload = extract_content_payload(report)
        job = {"mode": "summary_image", "content": payload, "reason": "missing_image"}
        image_queue.enqueue(job)
        enqueued += 1

    print(f"Scanned {scanned} report(s); enqueued {enqueued}; already-had-image {skipped_have}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from modules.render_api_client import RenderAPIClient
from modules.postgres_writer import PostgresWriter
from modules.report_generator import get_mp3_duration_seconds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("content_id", help="Content id (e.g., reddit:y5n87v or yt:<id>)")
    ap.add_argument("video_id", help="Stable video_id (e.g., y5n87v)")
    ap.add_argument("mp3_path", help="Path to local MP3 to (re)upload")
    args = ap.parse_args()

    base = os.getenv("RENDER_DASHBOARD_URL") or os.getenv("RENDER_API_URL")
    if not base:
        raise SystemExit("RENDER_DASHBOARD_URL or RENDER_API_URL is required")
    if not (os.getenv("INGEST_TOKEN") or os.getenv("SYNC_SECRET")):
        raise SystemExit("INGEST_TOKEN or SYNC_SECRET is required for uploads")

    mp3 = Path(args.mp3_path)
    if not mp3.exists():
        raise SystemExit(f"MP3 not found: {mp3}")

    client = RenderAPIClient(base_url=base)
    resp = client.upload_audio_file(mp3, args.content_id)
    print("upload_resp=", resp)

    # Derive root-relative url from response
    url: Optional[str] = None
    for k in ("public_url", "relative_path", "url", "path", "filename", "file"):
        v = resp.get(k)
        if isinstance(v, str) and v:
            url = v
            break
    if url and url.startswith("http"):
        from urllib.parse import urlparse
        p = urlparse(url)
        url = p.path
    print("chosen_url=", url)

    # Duration
    dur = get_mp3_duration_seconds(str(mp3)) or None

    pw = PostgresWriter()
    update = {
        "id": args.content_id,
        "video_id": args.video_id,
        "title": "temp",
        "canonical_url": "https://example.com",
        "thumbnail_url": "",
        "summary": {"summary": "tmp"},
        "media": {"has_audio": True},
        "audio_version": int(time.time()),
    }
    if url:
        update["media"]["audio_url"] = url
    if isinstance(dur, int) and dur > 0:
        update["media_metadata"] = {"mp3_duration_seconds": int(dur)}

    res = pw.upload_content(update)
    print("persist_result=", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


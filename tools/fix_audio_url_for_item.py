#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from pathlib import Path as _Path
import sys as _sys

# Ensure project root on sys.path for module imports when running as a script
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from modules.render_api_client import RenderAPIClient
from modules.report_generator import get_mp3_duration_seconds
import psycopg

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
    # If server returned a bare filename, assume /exports/audio/<filename>
    if url and '/' not in url:
        url = f"/exports/audio/{url}"
    print("chosen_url=", url)

    # Duration
    dur = get_mp3_duration_seconds(str(mp3)) or None

    version = int(time.time())
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required for database update")

    assignments = []
    params = []

    if url:
        assignments.append("media = coalesce(media, '{}'::jsonb) || jsonb_build_object('has_audio', true, 'audio_url', %s::text)")
        params.append(url)
    else:
        assignments.append("media = coalesce(media, '{}'::jsonb) || jsonb_build_object('has_audio', true)")

    if isinstance(dur, int) and dur > 0:
        assignments.append("media_metadata = coalesce(media_metadata, '{}'::jsonb) || jsonb_build_object('mp3_duration_seconds', %s::int)")
        params.append(int(dur))

    assignments.append("audio_version = %s::int")
    params.append(version)
    assignments.append("has_audio = true")

    params.append(args.video_id)

    sql = "UPDATE content SET " + ", ".join(assignments) + " WHERE video_id = %s"
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            print('rows_updated=', cur.rowcount)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

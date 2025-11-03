#!/usr/bin/env python3
"""
Backfill/verify audio flags and durations in Postgres.

Actions:
- For rows with media.has_audio=true, probe media.audio_url via HEAD (fallback Range GET).
  - On 404/missing: clear has_audio, remove audio_url, and unset mp3_duration_seconds.
  - On 200 and missing duration: ffprobe local file if available, else skip.

Flags:
  --dry-run                Do not write changes
  --limit N                Limit number of rows processed
  --where SQL              Additional SQL WHERE fragment
  --fix-duration-only      Only fill missing durations; do not clear flags

Environment:
  DATABASE_URL             Postgres DSN
  RENDER_DASHBOARD_URL     Base to resolve root-relative paths for HEAD checks
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Dict, Any
from urllib.parse import urljoin

import requests

try:
    import psycopg
except Exception as e:  # pragma: no cover
    print("psycopg not installed. Please add psycopg[binary]>=3.1 to requirements.", file=sys.stderr)
    raise


def head_or_range(url: str, timeout: int = 10) -> int:
    """Return HTTP status code for HEAD, falling back to Range GET 0-0."""
    try:
        r = requests.head(url, timeout=timeout)
        if r.status_code in (200, 204):
            return 200
        # Some proxies don't allow HEAD — fallback to Range GET
        headers = {"Range": "bytes=0-0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        return r.status_code
    except requests.RequestException:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--where", type=str, default="")
    ap.add_argument("--fix-duration-only", action="store_true")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    base = (os.getenv("RENDER_DASHBOARD_URL") or os.getenv("POSTGRES_DASHBOARD_URL") or os.getenv("RENDER_API_URL") or "").rstrip("/")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2

    limit_sql = f"LIMIT {int(args.limit)}" if args.limit and args.limit > 0 else ""
    where_extra = args.where.strip()

    cleared = 0
    filled = 0
    checked = 0
    errors = 0

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Discover columns present in 'content'
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'content'
                """
            )
            cols = {r[0] for r in cur.fetchall()}
            has_media = 'media' in cols
            has_media_md = 'media_metadata' in cols

            # Build a schema-aware SELECT
            select_parts = ["video_id"]
            select_parts.append("media" if has_media else "NULL AS media")
            select_parts.append("media_metadata" if has_media_md else "NULL AS media_metadata")

            base_where = "(has_audio IS TRUE)"
            if has_media:
                base_where = base_where + " OR ((media ->> 'has_audio')::boolean IS TRUE)"

            query = f"SELECT {', '.join(select_parts)} FROM content WHERE ({base_where})"
            if where_extra:
                query += f" AND ({where_extra})"
            query += f" ORDER BY updated_at DESC {limit_sql}"

            cur.execute(query)
            rows = cur.fetchall()
            import json as _json
            for (video_id, media, media_md) in rows:
                checked += 1
                if isinstance(media, str):
                    try:
                        media = _json.loads(media)
                    except Exception:
                        media = {}
                media = media or {}
                if isinstance(media_md, str):
                    try:
                        media_md = _json.loads(media_md)
                    except Exception:
                        media_md = {}
                media_md = media_md or {}
                audio_url = media.get("audio_url")

                # If missing audio_url, try canonical by_video path
                if not audio_url and video_id:
                    audio_url = f"/exports/by_video/{video_id}.mp3"

                # Resolve to absolute URL for probing
                abs_url = audio_url
                if audio_url and audio_url.startswith("/") and base:
                    abs_url = urljoin(base + "/", audio_url.lstrip("/"))

                status = 0
                if abs_url:
                    status = head_or_range(abs_url)

                if status == 200:
                    # Fill duration only if missing and requested
                    if (media_md.get("mp3_duration_seconds") in (None, 0)) and args.fix_duration_only:
                        # Duration requires local file; skip here to avoid NAS path coupling
                        pass
                else:
                    if args.fix_duration_only:
                        continue
                    # Clear flags
                    cleared += 1
                    if args.dry_run:
                        print(f"DRY-RUN clear: {video_id} url={audio_url} status={status}")
                    else:
                        try:
                            if has_media:
                                cur.execute(
                                    """
                                    UPDATE content
                                    SET media = jsonb_set(COALESCE(media, '{}'::jsonb) - 'audio_url', '{has_audio}', 'false'::jsonb, true),
                                        has_audio = FALSE,
                                        updated_at = now(),
                                        media_metadata = CASE WHEN media_metadata IS NULL THEN media_metadata ELSE (media_metadata - 'mp3_duration_seconds') END
                                    WHERE video_id = %s
                                    """,
                                    (video_id,),
                                )
                            else:
                                cur.execute(
                                    """
                                    UPDATE content
                                    SET has_audio = FALSE,
                                        updated_at = now()
                                    WHERE video_id = %s
                                    """,
                                    (video_id,),
                                )
                        except Exception:
                            errors += 1

            print({
                "checked": checked,
                "cleared": cleared,
                "duration_filled": filled,
                "errors": errors,
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

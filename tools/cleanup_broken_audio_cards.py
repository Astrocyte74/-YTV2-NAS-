#!/usr/bin/env python3
"""
Delete cards (content + summaries) in Postgres that claim to have audio
but do not have a playable MP3 available.

Criteria per row (video_id):
  - content.has_audio is TRUE, and
  - (media.audio_url is NULL/empty OR HEAD/Range GET to audio_url is not 200), and
  - If media.audio_url is missing, also probe fallback /exports/by_video/<video_id>.mp3
    and keep the row if that succeeds (safety against older layout).

Usage examples:
  # Dry-run on recent 200 items with has_audio=true
  python3 tools/cleanup_broken_audio_cards.py --dry-run --limit 200

  # Actually delete up to 100 broken rows
  python3 tools/cleanup_broken_audio_cards.py --limit 100

  # Scope further via a WHERE fragment (SQL AND ...)
  python3 tools/cleanup_broken_audio_cards.py --where "content_source IN ('reddit','youtube')" --limit 500

Environment:
  DATABASE_URL             Postgres DSN
  RENDER_DASHBOARD_URL     Base URL for HEAD checks (e.g., https://...onrender.com)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

try:
    import psycopg
except Exception as exc:  # pragma: no cover
    print("psycopg not installed. Please add psycopg[binary]>=3.1 to requirements.", file=sys.stderr)
    raise


def head_or_range(url: str, timeout: int = 10) -> int:
    try:
        r = requests.head(url, timeout=timeout)
        if r.status_code in (200, 204):
            return 200
    except requests.RequestException:
        pass
    # Some proxies disallow HEAD — try Range GET for a single byte
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout)
        if r.status_code in (200, 206):
            return 200
        return r.status_code
    except requests.RequestException:
        return 0


def normalize_json(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            import json as _json
            return _json.loads(v)
        except Exception:
            return {}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove broken audio cards from Postgres")
    ap.add_argument("--dry-run", action="store_true", help="Do not delete; just print what would be removed")
    ap.add_argument("--limit", type=int, default=200, help="Max rows to inspect (default 200)")
    ap.add_argument("--where", type=str, default="", help="Extra SQL WHERE fragment to AND")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2
    base = (os.getenv("RENDER_DASHBOARD_URL") or os.getenv("RENDER_API_URL") or "").rstrip("/")

    limit_sql = f"LIMIT {int(args.limit)}" if args.limit and args.limit > 0 else ""
    extra = f" AND ({args.where.strip()})" if args.where.strip() else ""

    candidates: List[Tuple[str, Optional[str]]] = []

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                (
                    "SELECT video_id, media FROM content "
                    "WHERE has_audio IS TRUE" + extra + " ORDER BY updated_at DESC " + limit_sql
                )
            )
            for vid, media in cur.fetchall():
                media_obj = normalize_json(media)
                audio_url = media_obj.get("audio_url")
                candidates.append((vid, audio_url))

    to_delete: List[str] = []
    checked = 0

    for vid, url in candidates:
        checked += 1
        # Resolve absolute
        def make_abs(u: Optional[str]) -> Optional[str]:
            if not u:
                return None
            if u.startswith("http"):
                return u
            if base and u.startswith("/"):
                return urljoin(base + "/", u.lstrip("/"))
            return None

        abs_url = make_abs(url)
        ok = False
        if abs_url:
            status = head_or_range(abs_url)
            ok = (status == 200)

        # Fallback by_video probe only if missing/failed
        if not ok:
            fallback = f"/exports/by_video/{vid}.mp3"
            abs_fb = make_abs(fallback)
            if abs_fb:
                status_fb = head_or_range(abs_fb)
                ok = (status_fb == 200)

        if not ok:
            to_delete.append(vid)

    if args.dry_run:
        print({"checked": checked, "to_delete_count": len(to_delete)})
        for vid in to_delete:
            print("DELETE", vid)
        return 0

    deleted = 0
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for vid in to_delete:
                cur.execute("DELETE FROM summaries WHERE video_id = %s", (vid,))
                cur.execute("DELETE FROM content WHERE video_id = %s", (vid,))
                deleted += 1

    print({"checked": checked, "deleted": deleted})
    for vid in to_delete:
        print("deleted", vid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


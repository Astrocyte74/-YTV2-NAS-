#!/usr/bin/env python3
"""
Remove audio variants from `summaries` where the corresponding `content` row
does not have a playable audio URL (i.e., content.has_audio=false and
media.audio_url is NULL/empty).

Why: Such rows lead the dashboard to render an audio icon without a playable
asset. We only want to keep `audio` variants when a public URL exists.

Usage:
  python3 tools/cleanup_audio_variants_no_url.py --dry-run
  python3 tools/cleanup_audio_variants_no_url.py            # performs deletion

Environment:
  DATABASE_URL (or DATABASE_URL_POSTGRES_NEW) must be set.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("psycopg is required") from exc


FIND_SQL = (
    """
    SELECT s.id, s.video_id
    FROM summaries s
    LEFT JOIN content c ON c.video_id = s.video_id
    WHERE s.variant = 'audio'
      AND (COALESCE(c.media->>'audio_url','') = '' OR c.media IS NULL)
      AND COALESCE(c.has_audio, FALSE) = FALSE
    ORDER BY s.id DESC
    """
)

DELETE_SQL = "DELETE FROM summaries WHERE id = ANY(%s)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cleanup orphan audio variants")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL_POSTGRES_NEW") or os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL or DATABASE_URL_POSTGRES_NEW is required", file=sys.stderr)
        return 2

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(FIND_SQL)
        rows: List[Tuple[int, str]] = cur.fetchall()

    if not rows:
        print({"to_delete": 0})
        return 0

    sample = rows[:20]
    print({"to_delete": len(rows)})
    for rid, vid in sample:
        print(f"DELETE id={rid} video_id={vid}")

    if args.dry_run:
        print("dry_run=True; no deletions performed")
        return 0

    ids = [rid for rid, _ in rows]
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(DELETE_SQL, (ids,))
        print({"deleted": cur.rowcount})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


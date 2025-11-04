#!/usr/bin/env python3
"""
Update one summary's HTML in Postgres by reformatting existing text with the
improved NAS formatter, inserting a new revision.

Usage:
  python3 tools/update_summary_html.py --video-id web:... --variant key-insights
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import psycopg

from modules.summary_variants import format_summary_html


def fetch_latest_text(conn: psycopg.Connection, video_id: str, variant: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text, revision
            FROM summaries
            WHERE video_id = %s AND variant = %s
            ORDER BY created_at DESC, revision DESC
            LIMIT 1
            """,
            (video_id, variant),
        )
        row = cur.fetchone()
        return row[0] if row and isinstance(row[0], str) else None


def next_revision(conn: psycopg.Connection, video_id: str, variant: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(MAX(revision), 0) FROM summaries WHERE video_id=%s AND variant=%s",
            (video_id, variant),
        )
        (rev,) = cur.fetchone()
        return int(rev) + 1


def insert_revision(conn: psycopg.Connection, video_id: str, variant: str, text: str, html: str, revision: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO summaries (video_id, variant, revision, text, html)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (video_id, variant, revision, text, html),
        )
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description="Reformat and insert a new HTML revision for one summary")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--variant", default="key-insights")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set")
        return 2

    with psycopg.connect(dsn) as conn:
        text = fetch_latest_text(conn, args.video_id, args.variant)
        if not text:
            print("No text found for", args.video_id, args.variant)
            return 1
        html = format_summary_html(text)
        rev = next_revision(conn, args.video_id, args.variant)
        insert_revision(conn, args.video_id, args.variant, text, html, rev)
        print("Inserted revision", rev, "for", args.video_id, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


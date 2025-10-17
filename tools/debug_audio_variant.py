#!/usr/bin/env python3
"""
Inspect audio-related fields for a given video_id in Postgres.

Usage:
    python tools/debug_audio_variant.py <video_id>

Example:
    python tools/debug_audio_variant.py lcqlgif2hFA
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "psycopg is not installed inside this environment. Run inside the container that has psycopg."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Postgres audio variant for a video_id.")
    parser.add_argument("video_id", help="Video ID (plain, e.g., lcqlgif2hFA)")
    parser.add_argument(
        "--dsn",
        help="Optional Postgres DSN. Defaults to DATABASE_URL environment variable.",
    )
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set and --dsn not provided.", file=sys.stderr)
        return 1

    video_id = args.video_id

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT has_audio FROM content WHERE video_id = %s", (video_id,))
            row = cur.fetchone()
            print(f"content.has_audio: {row[0] if row else None}")

            cur.execute(
                """
                SELECT variant, html, text, created_at
                FROM summaries
                WHERE video_id = %s
                ORDER BY created_at DESC
                """,
                (video_id,),
            )
            rows = cur.fetchall()

    if not rows:
        print("No summary rows found.")
        return 0

    print("\nSummary variants:")
    for variant, html, text, created_at in rows:
        html_snippet = (html[:100] + "…") if html else None
        text_snippet = (text[:100] + "…") if text else None
        print(f"- variant={variant!r} created_at={created_at}")
        print(f"  html={html_snippet}")
        print(f"  text={text_snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

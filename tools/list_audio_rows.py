#!/usr/bin/env python3
"""
List summary rows whose video_id contains a given substring.

Usage:
    python tools/list_audio_rows.py wAhTzwhh_WA
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("psycopg not installed inside this environment.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="List summary rows matching a video_id substring.")
    parser.add_argument("substring", help="Substring to look for in video_id (case sensitive)")
    parser.add_argument("--dsn", help="Optional Postgres DSN; defaults to DATABASE_URL env var.")
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set and --dsn not provided.", file=sys.stderr)
        return 1

    pattern = f"%{args.substring}%"

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, variant, html
                FROM summaries
                WHERE video_id LIKE %s
                ORDER BY video_id, variant
                """,
                (pattern,),
            )
            rows = cur.fetchall()

    if not rows:
        print("No rows found.")
        return 0

    for vid, variant, html in rows:
        snippet = (html[:80] + "…") if html else None
        print(f"{vid} :: {variant} :: {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

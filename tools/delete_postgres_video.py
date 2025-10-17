#!/usr/bin/env python3
"""
Delete a single video (content + summaries) from Postgres by video_id.

Usage:
    python tools/delete_postgres_video.py TEST1234567

Requires `psycopg[binary]>=3.1` and a DATABASE_URL environment variable.
"""

import argparse
import os
import sys

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "psycopg is not installed. Run `pip install 'psycopg[binary]>=3.1'` inside the container."
    ) from exc


def delete_video(video_id: str, dsn: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM summaries WHERE video_id = %s;", (video_id,))
            summaries_deleted = cur.rowcount or 0

            cur.execute("DELETE FROM content WHERE video_id = %s;", (video_id,))
            content_deleted = cur.rowcount or 0

        conn.commit()

    print(f"Deleted {summaries_deleted} summaries and {content_deleted} content rows for {video_id}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete a video from Postgres by video_id.")
    parser.add_argument("video_id", help="Video ID to delete (e.g. TEST1234567)")
    parser.add_argument(
        "--dsn",
        help="Optional Postgres DSN. Defaults to DATABASE_URL environment variable.",
    )
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: Set DATABASE_URL or pass --dsn.", file=sys.stderr)
        return 1

    return delete_video(args.video_id, dsn)


if __name__ == "__main__":
    raise SystemExit(main())

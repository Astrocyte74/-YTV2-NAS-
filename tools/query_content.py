#!/usr/bin/env python3
"""
Quick content query helper for Postgres.

Examples:
  python3 tools/query_content.py --title-like Dirac
  python3 tools/query_content.py --sql "SELECT id, video_id, title FROM content WHERE title ILIKE '%ART%' LIMIT 20"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

try:
    import psycopg
except Exception as e:  # pragma: no cover
    print("psycopg not installed. Please add psycopg[binary]>=3.1 to requirements.", file=sys.stderr)
    raise


def run_query(sql: str, params: List[str] | None = None) -> int:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        for r in rows:
            print(r)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", help="Raw SQL to run", default=None)
    ap.add_argument("--title-like", help="Title ILIKE filter (adds %% on both sides)", default=None)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if args.sql:
        return run_query(args.sql)

    if args.title_like:
        like = f"%{args.title_like}%"
        sql = (
            "SELECT id, video_id, title, has_audio FROM content "
            "WHERE title ILIKE %s ORDER BY updated_at DESC LIMIT %s"
        )
        return run_query(sql, [like, args.limit])

    print("Provide --sql or --title-like", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

try:
    import psycopg
except Exception:
    print("psycopg not installed. Please add psycopg[binary]>=3.1 to requirements.", file=sys.stderr)
    raise


def main() -> int:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2

    stmts = [
        "ALTER TABLE content ADD COLUMN IF NOT EXISTS media JSONB",
        "ALTER TABLE content ADD COLUMN IF NOT EXISTS media_metadata JSONB",
        "ALTER TABLE content ADD COLUMN IF NOT EXISTS audio_version INTEGER",
    ]

    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for sql in stmts:
            cur.execute(sql)
            print(f"OK: {sql}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


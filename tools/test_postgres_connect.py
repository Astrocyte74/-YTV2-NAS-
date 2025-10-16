#!/usr/bin/env python3
"""
Quick connectivity/permissions test for Postgres.

Usage:
  python tools/test_postgres_connect.py

Requires env:
  - DATABASE_URL or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE/PGSSLMODE
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import psycopg
except Exception as e:
    logger.error("psycopg not installed; pip install 'psycopg[binary]>=3.1'")
    sys.exit(1)


def main() -> int:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")
        user = os.getenv("PGUSER", "postgres")
        db = os.getenv("PGDATABASE", "postgres")
        logger.info(f"Connecting via PG* vars: host={host} port={port} user={user} db={db}")
    else:
        logger.info("Connecting via DATABASE_URL")

    try:
        with psycopg.connect(dsn or None) as conn:
            with conn.cursor() as cur:
                cur.execute("select version(), current_user")
                version, current_user = cur.fetchone()
                logger.info(f"Connected. version='{version}', user='{current_user}'")
                cur.execute("select now() at time zone 'utc'")
                now, = cur.fetchone()
                logger.info(f"UTC now: {now}")
        logger.info("✅ Success: basic queries ran")
        return 0
    except Exception as e:
        logger.error(f"❌ Connection or query failed: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


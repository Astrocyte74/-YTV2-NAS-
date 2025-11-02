#!/usr/bin/env python3
"""Clean summaries table entries with placeholder text (e.g., "Summary generation failed").

Usage:
    python tools/remove_placeholder_summaries.py --dry-run
    python tools/remove_placeholder_summaries.py            # performs deletion

Environment:
    DATABASE_URL_POSTGRES_NEW or DATABASE_URL must be set with write access.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List

import psycopg
from psycopg.rows import dict_row

LOGGER = logging.getLogger("remove_placeholder_summaries")


PLACEHOLDER_PREFIXES = (
    "summary generation failed",
    "unable to generate",
    "unable to create",
)

FIND_SQL = """
SELECT id, video_id, variant, created_at
FROM summaries
WHERE
    (
        lower(coalesce(text,'')) LIKE any(%(patterns)s)
        OR lower(coalesce(html,'')) LIKE any(%(patterns)s)
    )
ORDER BY created_at DESC;
"""

DELETE_SQL = "DELETE FROM summaries WHERE id = ANY(%s);"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove placeholder summaries")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matching rows; do not delete.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    dsn = os.getenv("DATABASE_URL_POSTGRES_NEW") or os.getenv("DATABASE_URL")
    if not dsn:
        LOGGER.error("DATABASE_URL_POSTGRES_NEW or DATABASE_URL must be set")
        return 1

    patterns = [prefix + "%" for prefix in PLACEHOLDER_PREFIXES]

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(FIND_SQL, {"patterns": patterns})
                rows = cur.fetchall()

            if not rows:
                LOGGER.info("No placeholder summaries found.")
                return 0

            LOGGER.info("Found %d placeholder summary rows:", len(rows))
            for row in rows[:20]:
                LOGGER.info(
                    "  id=%s video_id=%s variant=%s created_at=%s",
                    row["id"],
                    row["video_id"],
                    row["variant"],
                    row["created_at"],
                )
            if len(rows) > 20:
                LOGGER.info("  ... (%d more)", len(rows) - 20)

            if args.dry_run:
                LOGGER.info("Dry run – no deletions performed.")
                return 0

            ids: List[int] = [row["id"] for row in rows]
            with conn.cursor() as cur:
                cur.execute(DELETE_SQL, (ids,))
            conn.commit()

            LOGGER.info("Deleted %d rows.", len(ids))
            return 0

    except psycopg.Error as exc:  # pragma: no cover - defensive
        LOGGER.error("Database error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


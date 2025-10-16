#!/usr/bin/env python3
"""
Smoke test for Postgres upserts: inserts one content row and a couple of variants.

Usage:
  python tools/test_upsert_content.py [video_id]

Requires env:
  - DATABASE_URL or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE/PGSSLMODE
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from modules.postgres_writer import PostgresWriter
except Exception as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def sample_payload(video_id: str) -> dict:
    return {
        "id": f"yt:{video_id}",
        "video_id": video_id,
        "title": "Test Insert from NAS",
        "channel_name": "YTV2 Test Channel",
        "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "duration_seconds": 123,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "has_audio": False,
        "language": "en",
        "subcategories_json": {
            "categories": [
                {"category": "Technology", "subcategories": ["Programming & Software Development"]}
            ]
        },
        "summary": {
            "summary_type": "comprehensive",
            "summary": "This is a test summary.\n- Bullet A\n- Bullet B",
            "variants": [
                {"variant": "comprehensive", "text": "This is a test summary.\n- Bullet A\n- Bullet B"},
                {"variant": "key-insights", "text": "Insight 1 — why it matters."},
            ],
        },
    }


def main() -> int:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "TEST1234567"
    writer = PostgresWriter()

    if not writer.health_check():
        logger.error("Health check failed")
        return 2

    payload = sample_payload(video_id)
    result = writer.upload_content(payload)
    if not result:
        logger.error("Upsert failed")
        return 2
    logger.info(f"Upserted content for video_id={video_id}")
    logger.info("✅ Done. Verify on dashboard that card appears (HTML variant present).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


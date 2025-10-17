#!/usr/bin/env python3
"""
Strip leading 'yt:' prefixes from summaries.video_id rows (cleanup helper).

Usage:
    python tools/strip_yt_prefix_in_summaries.py
"""

import os
import sys

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("psycopg is required inside the container.") from exc


def main() -> int:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM summaries WHERE video_id LIKE 'yt:%'")
            count_before = cur.fetchone()[0]
            print(f"Rows with 'yt:' prefix before: {count_before}")

            if count_before:
                cur.execute(
                    """
                    UPDATE summaries AS tgt
                    SET video_id = SUBSTRING(video_id FROM 4)
                    WHERE video_id LIKE 'yt:%'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM summaries AS dup
                          WHERE dup.video_id = SUBSTRING(tgt.video_id FROM 4)
                            AND dup.variant = tgt.variant
                            AND dup.revision = tgt.revision
                      )
                    """
                )
                print(f"Updated rows: {cur.rowcount}")
        conn.commit()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT video_id, variant, revision FROM summaries WHERE video_id LIKE 'yt:%'")
            remaining = cur.fetchall()
            if remaining:
                print(f"Rows still prefixed (manual review needed): {len(remaining)}")
                for vid, variant, revision in remaining:
                    print(f"  - {vid} :: {variant} :: rev {revision}")
            else:
                print("Rows with 'yt:' prefix after: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Create the minimal Postgres tables/view/indexes the NAS needs.

Run this from the container (or any machine) that has psycopg installed and the
DATABASE_URL environment variable set. Example:

    python tools/setup_postgres_schema.py

You can override the DSN with --dsn and the grant target with --grant-user.
"""

import argparse
import os
import sys
from textwrap import dedent
from urllib.parse import urlparse

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "psycopg is not installed. Run `pip install 'psycopg[binary]>=3.1'` first."
    ) from exc


DDL_STATEMENTS = [
    dedent(
        """
        CREATE TABLE IF NOT EXISTS content (
          id TEXT PRIMARY KEY,
          video_id TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          channel_name TEXT,
          canonical_url TEXT,
          thumbnail_url TEXT,
          duration_seconds INTEGER,
          indexed_at TIMESTAMPTZ DEFAULT now(),
          updated_at TIMESTAMPTZ DEFAULT now(),
          has_audio BOOLEAN DEFAULT false,
          language TEXT,
          analysis_json JSONB,
          subcategories_json JSONB,
          topics_json JSONB
        );
        """
    ),
    # Ensure columns exist on older installations
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS channel_name TEXT;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS canonical_url TEXT;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS duration_seconds INTEGER;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ DEFAULT now();",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS has_audio BOOLEAN DEFAULT false;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS language TEXT;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS analysis_json JSONB;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS subcategories_json JSONB;",
    "ALTER TABLE content ADD COLUMN IF NOT EXISTS topics_json JSONB;",
    dedent(
        """
        CREATE TABLE IF NOT EXISTS summaries (
          id BIGSERIAL PRIMARY KEY,
          video_id TEXT NOT NULL,
          variant TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 1,
          text TEXT,
          html TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (video_id, variant, revision)
        );
        """
    ),
    # Ensure summary columns exist if table pre-dates current schema
    "ALTER TABLE summaries ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1;",
    "ALTER TABLE summaries ADD COLUMN IF NOT EXISTS text TEXT;",
    "ALTER TABLE summaries ADD COLUMN IF NOT EXISTS html TEXT;",
    "ALTER TABLE summaries ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();",
    "DROP VIEW IF EXISTS v_latest_summaries;",
    dedent(
        """
        CREATE OR REPLACE VIEW v_latest_summaries AS
        SELECT DISTINCT ON (video_id, variant)
          video_id, variant, text, html, created_at, revision
        FROM summaries
        WHERE (text IS NOT NULL OR html IS NOT NULL)
        ORDER BY video_id, variant, created_at DESC, revision DESC;
        """
    ),
    "CREATE INDEX IF NOT EXISTS idx_content_indexed_at ON content (indexed_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_content_lang ON content (language);",
    "CREATE INDEX IF NOT EXISTS idx_content_analysis_gin ON content USING GIN (analysis_json);",
    "CREATE INDEX IF NOT EXISTS idx_content_subcats_gin ON content USING GIN (subcategories_json);",
    "CREATE INDEX IF NOT EXISTS idx_summaries_lookup ON summaries (video_id, variant, created_at DESC);",
]


GRANT_TEMPLATE = dedent(
    """
    GRANT INSERT, UPDATE ON TABLE content, summaries TO {role};
    GRANT SELECT ON TABLE content, summaries, v_latest_summaries TO {role};
    """
)


def infer_default_role(dsn: str) -> str:
    parsed = urlparse(dsn)
    return parsed.username or "CURRENT_USER"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Postgres schema for YTV2.")
    parser.add_argument(
        "--dsn",
        help="Postgres DSN. Defaults to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--grant-user",
        help="Role to grant permissions to (defaults to DSN user). Use CURRENT_USER to grant to the connected role.",
    )
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: No DSN provided. Set DATABASE_URL or pass --dsn.", file=sys.stderr)
        return 1

    grant_role = args.grant_user or infer_default_role(dsn)

    print("Connecting to Postgres...")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for statement in DDL_STATEMENTS:
                print(f"Running:\n{statement.strip()}")
                try:
                    cur.execute(statement)
                except psycopg.Error as err:  # type: ignore[attr-defined]
                    print(f"❌ Error while running:\n{statement.strip()}\n{err}", file=sys.stderr)
                    conn.rollback()
                    return 1

            grant_sql = GRANT_TEMPLATE.format(role=grant_role)
            print(f"Applying grants to {grant_role}...")
            for stmt in grant_sql.strip().split(";\n"):
                if stmt:
                    try:
                        cur.execute(stmt + ";")
                    except psycopg.Error as err:  # type: ignore[attr-defined]
                        print(f"❌ Error applying grant `{stmt}`: {err}", file=sys.stderr)
                        conn.rollback()
                        return 1

        conn.commit()

    print("✅ Schema ensured. You can now run tools/test_upsert_content.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Apply follow-up research migration to PostgreSQL.

This script creates the follow_up_suggestions and follow_up_research_runs tables
required for the follow-up research feature.

Usage:
    python research_api/migrations/apply_001_follow_up_research.py

You can override the DSN with --dsn.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import psycopg
except Exception as exc:
    raise SystemExit(
        "psycopg is not installed. Run `pip install 'psycopg[binary]>=3.1'` first."
    ) from exc


MIGRATION_SQL = """
-- Migration 001: Add follow-up research tables
-- This migration adds support for storing follow-up research suggestions
-- and research run results for the YTV2 backend.

-- Table: follow_up_suggestions
-- Stores generated follow-up research suggestions for each summary.
-- This allows UI re-use of suggestions without re-generation.
CREATE TABLE IF NOT EXISTS follow_up_suggestions (
  id BIGSERIAL PRIMARY KEY,
  summary_id BIGINT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
  video_id TEXT NOT NULL,
  suggestions JSONB NOT NULL DEFAULT '[]'::jsonb,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  planner_provider TEXT NOT NULL DEFAULT 'unknown',
  planner_model TEXT NOT NULL DEFAULT 'unknown',
  source_context JSONB,
  -- Index for looking up suggestions by summary
  UNIQUE (summary_id)
);

-- Table: follow_up_research_runs
-- Stores the canonical source of truth for follow-up research runs.
-- Each run represents one coordinated execution of approved questions.
CREATE TABLE IF NOT EXISTS follow_up_research_runs (
  id BIGSERIAL PRIMARY KEY,
  summary_id BIGINT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
  video_id TEXT NOT NULL,

  -- User's approved questions and their metadata
  approved_questions TEXT[] NOT NULL,
  question_provenance TEXT[] NOT NULL,
  question_kinds TEXT[] NOT NULL,

  -- Planner's consolidated query plan
  planned_queries TEXT[] NOT NULL,
  coverage_map JSONB NOT NULL DEFAULT '[]'::jsonb,
  dedupe_notes TEXT NOT NULL DEFAULT '',

  -- Research execution settings
  provider_mode TEXT NOT NULL DEFAULT 'auto',
  tool_mode TEXT NOT NULL DEFAULT 'auto',
  depth TEXT NOT NULL DEFAULT 'balanced',
  compare BOOLEAN NOT NULL DEFAULT false,
  freshness_sensitive BOOLEAN NOT NULL DEFAULT false,

  -- LLM metadata
  planner_provider TEXT NOT NULL DEFAULT 'unknown',
  planner_model TEXT NOT NULL DEFAULT 'unknown',
  synth_provider TEXT NOT NULL DEFAULT 'unknown',
  synth_model TEXT NOT NULL DEFAULT 'unknown',

  -- Research results
  research_response TEXT NOT NULL,
  research_meta JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- Caching
  cache_key TEXT NOT NULL,

  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Prevent duplicate runs with same inputs
  UNIQUE (cache_key)
);

-- Create indexes for efficient lookups

-- Index for finding suggestions by summary
CREATE INDEX IF NOT EXISTS idx_follow_up_suggestions_summary_id
  ON follow_up_suggestions (summary_id);

-- Index for finding suggestions by video
CREATE INDEX IF NOT EXISTS idx_follow_up_suggestions_video_id
  ON follow_up_suggestions (video_id);

-- Index for finding research runs by summary
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_summary_id
  ON follow_up_research_runs (summary_id);

-- Index for finding research runs by video
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_video_id
  ON follow_up_research_runs (video_id);

-- Index for finding research runs by cache key (cache lookup)
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_cache_key
  ON follow_up_research_runs (cache_key);

-- Index for finding recent research runs
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_created_at
  ON follow_up_research_runs (created_at DESC);

-- GIN index for querying approved_questions array
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_approved_questions
  ON follow_up_research_runs USING GIN (approved_questions);

-- GIN index for querying research_meta JSONB
CREATE INDEX IF NOT EXISTS idx_follow_up_research_runs_meta_gin
  ON follow_up_research_runs USING GIN (research_meta);

-- Add follow_up_research_available column to summaries table
-- This is a lightweight UI flag that references the runs table
ALTER TABLE summaries ADD COLUMN IF NOT EXISTS follow_up_research_available BOOLEAN NOT NULL DEFAULT false;

-- Create index for summaries with available follow-up research
CREATE INDEX IF NOT EXISTS idx_summaries_follow_up_available
  ON summaries (follow_up_research_available)
  WHERE follow_up_research_available = true;
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply follow-up research migration (001) to PostgreSQL."
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN. Defaults to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing.",
    )
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: No DSN provided. Set DATABASE_URL or pass --dsn.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("DRY RUN - Would execute the following SQL:")
        print(MIGRATION_SQL)
        return 0

    print("Connecting to Postgres...")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            print("Applying migration 001: Add follow-up research tables...")

            # Split by semicolon and execute each statement
            statements = [s.strip() for s in MIGRATION_SQL.split(";") if s.strip()]

            for i, statement in enumerate(statements, 1):
                if not statement:
                    continue
                # Skip comments
                if statement.strip().startswith("--"):
                    continue

                # Print first line of statement for progress
                first_line = statement.split("\n")[0][:60]
                print(f"[{i}/{len(statements)}] {first_line}...")

                try:
                    cur.execute(statement)
                except psycopg.Error as err:
                    print(f"❌ Error while running:\n{statement[:200]}...\n{err}", file=sys.stderr)
                    conn.rollback()
                    return 1

        conn.commit()

    print("✅ Migration 001 applied successfully!")
    print("\nTables created:")
    print("  - follow_up_suggestions")
    print("  - follow_up_research_runs")
    print("\nIndexes created:")
    print("  - idx_follow_up_suggestions_summary_id")
    print("  - idx_follow_up_suggestions_video_id")
    print("  - idx_follow_up_research_runs_summary_id")
    print("  - idx_follow_up_research_runs_video_id")
    print("  - idx_follow_up_research_runs_cache_key")
    print("  - idx_follow_up_research_runs_created_at")
    print("  - idx_follow_up_research_runs_approved_questions (GIN)")
    print("  - idx_follow_up_research_runs_meta_gin (GIN)")
    print("  - idx_summaries_follow_up_available")
    print("\nColumns added:")
    print("  - summaries.follow_up_research_available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

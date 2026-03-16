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
  expires_at TIMESTAMPTZ,
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
  question_provenance TEXT[] NOT NULL,  -- e.g., ["suggested", "preset", "custom"]
  question_kinds TEXT[] NOT NULL,        -- e.g., ["pricing", "comparison", "what_changed"]

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
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'ok',

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

ALTER TABLE follow_up_suggestions ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
ALTER TABLE follow_up_research_runs ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;
ALTER TABLE follow_up_research_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok';

-- Create index for summaries with available follow-up research
CREATE INDEX IF NOT EXISTS idx_summaries_follow_up_available
  ON summaries (follow_up_research_available)
  WHERE follow_up_research_available = true;

-- Add comment documentation
COMMENT ON TABLE follow_up_suggestions IS 'Stores generated follow-up research suggestions for each summary. Allows UI re-use without re-generation.';
COMMENT ON TABLE follow_up_research_runs IS 'Canonical source of truth for follow-up research runs. Each run represents one coordinated execution of approved questions.';
COMMENT ON COLUMN follow_up_suggestions.suggestions IS 'JSONB array of suggestion objects with id, label, question, reason, kind, priority, default_selected.';
COMMENT ON COLUMN follow_up_research_runs.cache_key IS 'Unique cache key: video_id + summary_id + normalized_questions + provider_mode + depth';
COMMENT ON COLUMN follow_up_research_runs.approved_questions IS 'User-approved research directions (max 3)';
COMMENT ON COLUMN follow_up_research_runs.question_provenance IS 'Where each question came from: suggested, preset, or custom';
COMMENT ON COLUMN follow_up_research_runs.question_kinds IS 'Question category: pricing, comparison, what_changed, background, etc.';
COMMENT ON COLUMN follow_up_research_runs.planned_queries IS 'Consolidated search queries from planner (deduped)';
COMMENT ON COLUMN follow_up_research_runs.coverage_map IS 'Maps approved questions to planned queries showing coverage';
COMMENT ON COLUMN summaries.follow_up_research_available IS 'Lightweight UI flag; check follow_up_research_runs table for actual data';

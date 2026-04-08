-- Migration 002: Add follow-up chat turns table
-- Persists grounded follow-up chat exchanges linked to deep-research runs.
-- Each row is one user question / assistant answer pair.

CREATE TABLE IF NOT EXISTS follow_up_chat_turns (
  id BIGSERIAL PRIMARY KEY,
  follow_up_run_id BIGINT NOT NULL REFERENCES follow_up_research_runs(id) ON DELETE CASCADE,
  video_id TEXT NOT NULL,
  summary_id BIGINT NULL REFERENCES summaries(id) ON DELETE SET NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  chat_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_follow_up_chat_turns_run_id
  ON follow_up_chat_turns (follow_up_run_id);

CREATE INDEX IF NOT EXISTS idx_follow_up_chat_turns_video_id
  ON follow_up_chat_turns (video_id);

CREATE INDEX IF NOT EXISTS idx_follow_up_chat_turns_created_at
  ON follow_up_chat_turns (created_at DESC);

-- Comments
COMMENT ON TABLE follow_up_chat_turns IS 'Persisted grounded follow-up chat exchanges over deep-research reports. Each row is one user question + assistant answer pair.';
COMMENT ON COLUMN follow_up_chat_turns.follow_up_run_id IS 'FK to the deep-research run this chat turn belongs to. CASCADE delete when the run is removed.';
COMMENT ON COLUMN follow_up_chat_turns.chat_meta IS 'Lightweight metadata: llm_provider, llm_model, mode, source_count, thread_turn_count_at_answer_time.';

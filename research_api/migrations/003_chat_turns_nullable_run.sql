-- Migration 003: Allow pre-research chat persistence
-- Makes follow_up_run_id nullable so chat turns can be stored
-- before a deep research run exists.

ALTER TABLE follow_up_chat_turns
  ALTER COLUMN follow_up_run_id DROP NOT NULL;

-- Add a partial index for fast lookups by video_id only (pre-research turns)
CREATE INDEX IF NOT EXISTS idx_follow_up_chat_turns_video_only
  ON follow_up_chat_turns (video_id, created_at DESC)
  WHERE follow_up_run_id IS NULL;

COMMENT ON COLUMN follow_up_chat_turns.follow_up_run_id IS 'FK to the deep-research run this chat turn belongs to. NULL for pre-research chat (before any deep research run exists).';

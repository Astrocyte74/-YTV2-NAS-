# NAS TTS Integration Overview

## Current Components

- `YouTubeSummarizer.generate_tts_audio(...)`
  - Acts as the single entry point for audio generation.
  - Supports two providers: `local` (WireGuard-accessible TTS hub) and `openai` (cloud fallback).
  - Retains chunking + stitching for long texts.

- `modules/tts_hub.py`
  - Async client for the local TTS hub (`/api/meta`, `/voices_catalog`, `/favorites`, `/synthesise`).
  - Exposes accent-family helpers (`available_accent_families`, `filter_catalog_voices`).
  - Raises `LocalTTSUnavailable` when the hub cannot be reached.

- `modules/tts_queue.py`
  - Lightweight file-based queue (`data/tts_queue/`).
  - Stores jobs for deferred local synthesis when the hub is offline.

- `modules/telegram_handler.py`
  - Orchestrates Telegram prompts.
  - For `/tts` and audio summaries it asks: “Local hub vs OpenAI?”.
  - On local failure offers: “Queue for later” or “Use OpenAI”.
  - Reuses the catalog UI (gender → accent family → voices) for selections.

## Typical Flow

1. User requests `/tts` or triggers audio summary.
2. Prompt for provider:
   - Local hub → attempt immediate synthesis.
   - OpenAI → call `generate_tts_audio(..., provider="openai")`.
   - Local offline → display second prompt (queue vs OpenAI).
3. If queued, job is written to `data/tts_queue/` for later replay.
4. Successful synthesis delivers audio via Telegram and pushes to targets (SQLite/Postgres, render).

## Future Considerations

- Build a queue worker to drain `data/tts_queue/` when the hub comes online.
- Add CLI or dashboard controls for queued jobs (view, retry, purge).
- Extract provider selection from Telegram into a reusable function for CLI/batch jobs.
- Expose config for default voices/accents (currently hard-coded fallback voice when using OpenAI).
- Consider caching hub catalog responses to avoid fetching on every `/tts` invocation.

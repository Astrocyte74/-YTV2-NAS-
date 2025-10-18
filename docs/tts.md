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
  - Orchestrates Telegram prompts and UI for both one‑off TTS (`/tts`) and YouTube summary audio.
  - Provider picker: “Local TTS hub” vs “OpenAI TTS”. On local failure: “Queue for later” or “Use OpenAI”.
  - Voice picker: Favorites‑first by default (if any). Toggle between Favorites and All voices. Gender and accent‑family filters supported.
  - UX details: Shows a status bubble when a voice is selected (⏳ Generating … → ✅ Generated …), keeps the picker open for rapid A/B tests, and includes the voice name in audio captions.

## Typical Flow

1. User requests either:
   - One‑off preview: `/tts your text here` (no DB/Render sync), or
   - Summary audio: choose an audio variant from the YouTube flow (DB/Render sync enabled).
2. The bot prompts for provider:
   - Local hub → synthesize via the NAS TTS hub (preferred).
   - OpenAI → synthesize via OpenAI as fallback.
   - If local hub is offline/unreachable → offer to queue job or switch to OpenAI.
3. Voice selection:
   - Favorites are shown/selected by default when available (falls back to global favorites if `tag=telegram` is empty).
   - Optionally switch to All voices; filter by gender and accent family.
4. When you click a voice, the bot posts a “Generating …” status bubble, then replies with the audio and updates the bubble to “Generated …”. The picker remains open for more tests.
5. Delivery and sync:
   - One‑off preview (`/tts`): sends audio with a compact caption and does NOT sync to DB/Render.
   - Summary audio (YouTube flow): sends audio and syncs to Postgres + uploads to Render, updating dashboard playback.

## Modes and Captions

- One‑off TTS (`/tts`)
  - Mode key: `oneoff_tts`
  - Caption: `TTS Preview • {voice} • {provider}`
  - No DB/Render upload, intended for quick voice auditioning. Picker stays open after each run.

- Summary Audio (YouTube flow)
  - Mode key: `summary_audio`
  - Caption: `Audio Summary: {Title} • {voice}` on the first line; provider on the next line.
  - DB/Render upload enabled (controls appear on dashboard; Listen chips stream the saved MP3).

## Favorites Resolution and Defaults

- When provider `local` is selected, the voice picker defaults to Favorites if any favorites exist.
- Favorites are fetched with `tag=telegram`; if none found, falls back to global `favorites`.
- The picker checkmarks reflect the current mode: “Favorites” or “All voices”.

## Environment

- Local hub base URL: `TTSHUB_API_BASE` (required for `local` provider)
- OpenAI key: `OPENAI_API_KEY` (required for `openai` provider)
- Dashboard sync (summary audio only):
  - `DATABASE_URL` – Postgres connection for direct upserts
  - `AUDIO_PUBLIC_BASE` – Base used to construct public audio URLs (e.g., `https://your-host` → `${AUDIO_PUBLIC_BASE}/exports/<file>.mp3`)

## Future Considerations

- Build a queue worker to drain `data/tts_queue/` when the hub comes online.
- Add CLI or dashboard controls for queued jobs (view, retry, purge).
- Extract provider selection from Telegram into a reusable function for CLI/batch jobs.
- Expose config for default voices/accents (currently hard-coded fallback voice when using OpenAI).
- Consider caching hub catalog responses to avoid fetching on every `/tts` invocation.
 - Optional env toggles to select between Minimal (favorites‑only), Compact (reduced header), and Full catalog UI for `/tts`.
 - Add a simple “Close” button to dismiss the picker explicitly when testing is done.

## Troubleshooting

- Clicks don’t register
  - Ensure the picker was created after the latest restart (old keyboards won’t use compact callback keys).
  - Check logs for `🔔 Callback received:` and `🎛️ TTS callback:` lines.

- Local hub unreachable
  - Confirm `TTSHUB_API_BASE` is set in the container env and that the NAS can reach the hub URL.
  - Use the fallback prompt to queue or switch to OpenAI.

- Favorites toggle does nothing
  - If `tag=telegram` favorites are empty, the picker falls back to global favorites. Toggle should then reflect “Favorites” mode with that list.

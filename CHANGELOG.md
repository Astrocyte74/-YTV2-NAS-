# Changelog

## 2025-11-04 — Linear audio flow, auto‑defaults, and richer status

Highlights
- Linear audio summary flow for all sources (YouTube, Reddit, Web): LLM → summary + illustration → TTS.
- Auto‑defaults with countdown hints:
  - LLM auto‑default (audio): after `LLM_AUTO_DEFAULT_SECONDS`, prefer `QUICK_LOCAL_MODEL`, else `QUICK_CLOUD_MODEL`.
  - TTS auto‑default: after `TTS_AUTO_DEFAULT_SECONDS`, prefer first `TTS_QUICK_FAVORITE` (Local), else `TTS_CLOUD_VOICE` (OpenAI).
- TTS auto‑select now drives the same manual selection code path; fixes cases where hub POST didn’t occur.
- Richer status during summary (spinner + step cues) and distinct “Generating illustration…” cue before images.

User‑Facing Changes
- After choosing Audio Summary, you may see an LLM chooser; if no action, the bot auto‑picks your configured default with a visible countdown and starts.
- After summary/illustration, the TTS chooser appears; if no action, it auto‑selects and the message updates to “Starting text‑to‑speech …” while the keyboard disappears.
- Clear progress during TTS (preparing → chunk i/N → combining → upload) and during summary (drafting/translate/analyze + spinner).

Environment
- LLM auto‑default: `LLM_AUTO_DEFAULT_SECONDS`, `QUICK_LOCAL_MODEL`, `QUICK_CLOUD_MODEL`
- TTS auto‑default: `TTS_AUTO_DEFAULT_SECONDS`, `TTS_QUICK_FAVORITE`, `TTS_CLOUD_VOICE`, optional `TTS_FALLBACK_TO_OPENAI`
- Status cadence: `SUMMARY_STATUS_INTERVAL` (default 10s)

Reliability
- TTS auto‑select invokes the same handler as manual selection (no stale callback issues); local synth uses provider‑aware chunking; optional OpenAI fallback on timeout.

## 2025-11-03 — Robust audio uploads, JSON enrichment, and cleanup tools

Highlights
- Dashboard JSON now enriches audio variants for both list and single views (`audio_url + duration`); `has_audio` mirrors list semantics.
- Upload handlers hardened: streaming multipart parsing, atomic temp→rename writes, and clear error logging; support `Authorization: Bearer` and `X-INGEST-TOKEN`.
- NAS only emits audio variants when a playable `audio_url` exists; removed legacy placeholders to prevent “audio icon only” cards.
- Backfill and cleanup tools added for rapid remediation.

User‑Facing Changes
- Listen chips appear only when a real MP3 is present; older placeholder cards were cleaned up.
- JSON endpoints return enriched `summary_variants` with audio entries and accurate `has_audio`.

Technical Changes
- NAS: emit audio ingest variants only when `audio_url` is present; strip placeholder entries.
- NAS tools:
  - `tools/batch_fix_audio_urls.py`
  - `tools/scan_and_fix_from_exports.py`
  - `tools/cleanup_audio_variants_no_url.py`
  - `tools/cleanup_broken_audio_cards.py`
- Dashboard:
  - `/api/upload-audio` and `/api/upload-image` now accept Bearer or X-INGEST-TOKEN; parse streaming; write atomically; return `public_url`/`size`.
  - List and single JSON use the same enrichment path.

Operational Notes
- Ensure sufficient free space on the Render disk mounted at `/app/data`; uploads fail with 500 and won’t write partial files. Use HEAD to validate `Content-Length > 0`.
- Prefer server‑returned `public_url` and treat size==0 as failure.

Verification
- Upload: `POST /api/upload-audio` → 200 JSON with `public_url` and `size > 0`.
- HEAD: `/exports/audio/<filename>.mp3?v=<audio_version>` → 200, non‑zero size.
- JSON: `/<id>.json` shows `has_audio:true` and audio variant with `audio_url + duration`.

## 2025-10-18 — TTS provider selection, improved picker UX, favorites default, and docs

Highlights
- Provider choice: Local TTS hub vs OpenAI with graceful fallback
- Favorites-first voice picker with working Favorites/All toggle
- Clear click feedback (status bubble) and audio captions include voice
- One-off /tts flow stays compact and keeps the picker open
- Documentation added for current behavior and troubleshooting

User‑Facing Changes
- /tts (one-off preview)
  - Shows status bubble: “⏳ Generating …” → “✅ Generated …”
  - Sends audio with “TTS Preview • {voice} • {provider}”
  - Picker remains open for rapid A/B testing; no DB/Render sync
- YouTube summary audio
  - Defaults to Favorites if available; toggle Favorites/All; filter by gender/accent
  - Audio captions include title + voice; normal DB/Render sync
  - Picker stays open after each generation

Technical Changes
- Add shared local TTS client with helpers
  - `modules/tts_hub.py`
- Add simple file‑backed job queue for local fallback
  - `modules/tts_queue.py`
- Update Telegram handler (provider UI, catalog UI, status bubble, captions, open picker)
  - `modules/telegram_handler.py`
- Extend summarizer TTS to support local/OpenAI + favorites
  - `youtube_summarizer.py`
- Add docs for flow, modes, env, and troubleshooting
  - `docs/tts.md`

Configuration
- Local TTS hub: `TTSHUB_API_BASE` (required for local provider)
- OpenAI fallback: `OPENAI_API_KEY`
- Summary audio sync (YouTube flow only):
  - `DATABASE_URL` and `AUDIO_PUBLIC_BASE`

Compatibility
- No breaking changes to existing summarization
- /tts now has a richer UI; defaults remain simple for favorites flow
- Favorites default for catalog flows; automatically flips to all voices when no favorites match the active engine
- Engine toggle chips surface when multiple providers (Kokoro/XTTS, etc.) are available, and the picker auto-switches to the first engine that contains your favorites
- Added an `ALL` engine tab and per-voice engine prefixes (e.g., `[K]`, `[X]`) so cross-engine favorites are obvious and accessible in one list

Deployment
- Pull latest `main` and restart the bot container
- Ensure env vars are set: `TTSHUB_API_BASE`, `OPENAI_API_KEY`, `DATABASE_URL`, `AUDIO_PUBLIC_BASE` (as applicable)

Verification
- One‑off test: `/tts hello world` → Local → select a voice
  - Expect status bubble + voice caption; picker remains open
- Summary audio test: run audio variant from YouTube flow → Local → Favorites voice
  - Expect status bubble + voice caption + DB/Render sync
- Logs to confirm flow:
  - “🔔 Callback received …”, “🎛️ TTS callback …”, “📦 TTS file ready …”

Known Issues / Next Steps
- Optional env toggle for Minimal vs Compact vs Full UI for `/tts`
- Worker to drain `data/tts_queue/` when hub comes back online
- Short‑TTL caching for catalog/favorites to reduce HTTP calls
- “Close” button to dismiss the picker when testing is done

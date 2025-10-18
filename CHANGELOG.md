# Changelog

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
- Favorites default for catalog flows; falls back to global favorites when `tag=telegram` is empty

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


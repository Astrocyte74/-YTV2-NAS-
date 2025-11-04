# YTV2-NAS Operational Notes

## Telegram Bot Content Flow
- Unified pipeline supports YouTube videos, Reddit threads, and generic web articles (layered extractor with Readability/Trafilatura fallbacks).
- Telegram stores the active item context (`source`, `url`, `content_id`, etc.) so all summary types reuse the same keyboard.
- Summaries are exported locally (JSON optional) and written directly to Postgres via UPSERTs; ledger keys use universal IDs (`yt:<id>`, `reddit:<id>`).
- Audio variants are generated on the NAS, flagged in Postgres (`content.has_audio=true`), and then pulled by the dashboard from the synced exports share so Listen chips stream immediately.

### Audio Summary (Linear Flow)
- Linear sequence for all sources (YouTube, Reddit, Web):
  1) Choose “Audio Summary”
  2) LLM selection (auto‑default supported)
  3) Summary + illustration
  4) TTS selection (auto‑default supported) → MP3 + dashboard upload

- LLM auto‑default (optional):
  - If `LLM_AUTO_DEFAULT_SECONDS > 0`, the chooser shows a countdown hint and auto‑selects after the delay:
    - Prefer `QUICK_LOCAL_MODEL` (e.g., `gemma3:12b`), else fall back to `QUICK_CLOUD_MODEL` (e.g., `google/gemini-2.5-flash-lite`).
  - When auto‑select triggers, the chooser message updates to “Starting summary …” and the keyboard is removed.

- TTS auto‑default (optional):
  - If `TTS_AUTO_DEFAULT_SECONDS > 0`, the TTS chooser shows a countdown hint and auto‑selects after the delay:
    - Prefer Local favorite (first from `TTS_QUICK_FAVORITE`, e.g., `kokoro|favorite--bm-daniel`), else fall back to OpenAI voice `TTS_CLOUD_VOICE` (e.g., `fable`).
  - When TTS starts (auto or manual), the chooser updates to “Starting text‑to‑speech …” and the keyboard is removed. Status then shows fine‑grained TTS progress (preparing → chunk i/N → combining → upload) before the MP3 posts.

- Summary status updates (before illustration):
  - Periodic spinner with elapsed time (`SUMMARY_STATUS_INTERVAL`, default 10s): “🔄/⏳/⌛ Analyzing content and drafting summary… (Xs)”
  - Step‑level cues (lightweight): drafting audio summary text, translating (fr/es), analyzing & categorizing
  - Illustration cue: “🎨 Generating illustration…” just before the image is attached

## Draw Things Integration
- Hub base: set `TTSHUB_API_BASE=http://192.168.7.134:7860/api` (WireGuard IP + hub port). All calls go through the hub proxy; do not hit Draw Things on `127.0.0.1:7861` from the NAS.
- Convenience endpoint: `POST $TTSHUB_API_BASE/telegram/draw` accepts `{prompt,width,height,steps,seed?,negative?,sampler?,cfgScale?}` and returns a JSON payload with a relative `url` under `/image/drawthings/`.
- Quick probes:  
  - `curl -sS "$TTSHUB_API_BASE/meta" | jq`  
  - `curl -sS "$TTSHUB_API_BASE/drawthings/models" | jq`  
  - `curl -sS -X POST "$TTSHUB_API_BASE/telegram/draw" -H 'content-type: application/json' -d '{"prompt":"Sunlit watercolor fox","steps":20,"width":512,"height":512}' | jq`
- Model catalogue: set `DRAW_MODELS` to control the Telegram picker (comma-separated `name:group`, flux/general). Defaults are `Flux.1 [schnell]:flux,HiDream I1 fast:general`. Flux-tagged models surface Flux presets; everything else uses the general preset family.
- Preset catalogue: `GET $TTSHUB_API_BASE/telegram/presets` returns presets, style presets, negative presets, and defaults (`flux_balanced` by default). The bot caches this so the UI can stay in sync with hub changes.
- Telegram command: `/draw <prompt>` (`/d`) opens a menu to enhance the prompt via Local (Ollama) or Cloud LLMs, choose a preset/style/negative combination, and generate images using the preset’s size or manual Small/Medium/Large overrides. While a task runs, the inline keyboard collapses to a disabled “Working…” state, status text appears under the prompt, and the bot uploads the generated image bytes directly to Telegram (so LAN-only URLs are fine). All steps log under the `draw:` prefix in `bot.log`.
- Note: some Draw Things builds return 404 for `sd-models`/`samplers`; the hub now translates that to `200 []`. Use the convenience endpoint to validate generation rather than the raw list calls.

## Reddit Integration
- Credentials required: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (blank for installed app), `REDDIT_REFRESH_TOKEN`, `REDDIT_USER_AGENT`.
- Fetcher handles canonical URLs, `redd.it/<id>`, and the short `/r/<sub>/s/<token>` share links.
- JSON reports include `content_source="reddit"`, subreddit metadata, and comment snippets for context.

## Deployment & Container Tips
- Python dependencies (PRAW, psycopg, etc.) are installed via `requirements.txt`. Any time the file changes, rebuild the NAS image so the packages are baked in:
  1. From the repo root, build with an explicit, date-stamped tag (recommended): `docker build -t ytv2-nas:web-ingest-watcher-2025.10.22-extras .`
  2. In Portainer: **Containers → youtube-summarizer-bot → Duplicate/Edit → Replace existing container**, set Image to `ytv2-nas:web-ingest-watcher-2025.10.22-extras` (or your latest tag), confirm env vars, then deploy.
  3. After the new container starts, run `python tools/test_postgres_connect.py` from the console to confirm psycopg is available.
- Portainer: Use **Recreate** with _Re-pull image_ **off** to pick up code changes from the bind-mounted repo.
- After updating `.env.nas` (e.g., new Reddit refresh token), recreate the container so env vars reload.
- Helper scripts:
  - `tools/setup_postgres_schema.py` — schema/grants bootstrapping
  - `tools/debug_audio_variant.py` — inspect `content.has_audio` and `<audio>` HTML served from Postgres exports
  - `tools/test_reddit_connection.py`, `tools/debug_reddit_env.py` — Reddit diagnostics

## Operational Toggles
- YouTube access switch: set `YOUTUBE_ACCESS` to `1|true|yes|on` to enable handling of YouTube links; any other value pauses YouTube summaries (Telegram replies with a friendly pause message).
- yt-dlp mitigation and pacing (honored by the Python code and surfaced in `docker-compose.yml`):
  - `YTDLP_SAFE_MODE` — prefer resilient client profile (e.g., `true`)
  - `YTDLP_FORCE_CLIENT` — `android|web|tv|web_safari`
  - `YTDLP_SLEEP_REQUESTS` — seconds between requests (e.g., `3`)
  - `YTDLP_RETRIES` — total retries (applies to fragments as well)
  - `YTDLP_COOKIES_FILE` — path to a Netscape cookies file
  - `YTDLP_FORCE_STACK` — `ipv4|ipv6` (maps to source address)
  - `YT_DLP_OPTS` — raw flags appended to yt-dlp calls

- Fast reachability probes (preflight checks; keep responsive UX when Mac is offline):
  - `REACH_CONNECT_TIMEOUT` — TCP connect timeout in seconds (default `2`)
  - `REACH_HTTP_TIMEOUT` — HTTP read timeout in seconds (default `4`)
  - `REACH_TTL_SECONDS` — cache TTL for last probe result (default `20`)
  - Used by `/o` and `/tts` to quickly detect when the hub or Ollama is unreachable and show an immediate message instead of waiting on long timeouts.

- TTS hub timeouts (fine‑tune request timeouts to the hub):
  - `TTSHUB_TIMEOUT_CATALOG` — voices catalog fetch (default `8`)
  - `TTSHUB_TIMEOUT_FAVORITES` — favorites fetch (default `6`)
  - `TTSHUB_TIMEOUT_SYNTH` — synthesis POST (default `20`)
  - These apply only to TTS requests and can be left at defaults in most setups.

- Linear Audio Flow (auto‑defaults & status):
  - `LLM_AUTO_DEFAULT_SECONDS` — auto‑select the LLM after N seconds (audio only)
  - `QUICK_LOCAL_MODEL` — default local model for LLM auto‑select (e.g., `gemma3:12b`)
  - `QUICK_CLOUD_MODEL` — default cloud model (slug, e.g., `google/gemini-2.5-flash-lite`)
  - `TTS_AUTO_DEFAULT_SECONDS` — auto‑select TTS after N seconds
  - `TTS_QUICK_FAVORITE` — comma‑separated local favorites (`engine|slug`) for auto‑default
  - `TTS_CLOUD_VOICE` — OpenAI voice fallback (e.g., `fable`)
  - `TTS_FALLBACK_TO_OPENAI` — `1|true|yes` enables auto‑fallback on local failure/timeout
  - `SUMMARY_STATUS_INTERVAL` — seconds between status spinner updates during summary (default `10`)

## Admin & Dashboard Health

- Dashboard version: `GET /api/version` (no auth)
- Storage health (gated): `GET /api/health/storage` with `Authorization: Bearer $DASHBOARD_DEBUG_TOKEN`
- The `/status` command on NAS shows version + storage when the token is set.
- Early storage gate in chat:
  - Warns at ≥90% and ≥95% usage
  - Blocks new processing at ≥99%

## Dashboard Notes (Postgres + Uploads)
- Dashboard reads from Postgres and accepts authenticated uploads for audio/images.
- JSON endpoints (`/api/reports`, `/<id>.json`) enrich `summary_variants` with `{ kind:'audio', audio_url, duration }` based on `content.media`/`media_metadata`.
- Ensure:
  - At least one text variant has non‑null `html` (for text cards), or
  - A playable audio variant exists (NAS wrote `media.audio_url` and `mp3_duration_seconds`).
- Audio uploads: `POST /api/upload-audio` (Bearer or X‑INGEST‑TOKEN). Fallback: `POST /ingest/audio`.
- Health: `GET /health/ingest` reports `token_set` and `pg_dsn_set`.

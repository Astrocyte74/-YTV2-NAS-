# YTV2-NAS Operational Notes

## Telegram Bot Content Flow
- Unified pipeline supports YouTube videos, Reddit threads, and generic web articles (layered extractor with Readability/Trafilatura fallbacks).
- Telegram stores the active item context (`source`, `url`, `content_id`, etc.) so all summary types reuse the same keyboard.
- Summaries are exported locally (JSON optional) and written directly to Postgres via UPSERTs; ledger keys use universal IDs (`yt:<id>`, `reddit:<id>`).
 - Audio variants are generated on the NAS, flagged in Postgres (`content.has_audio=true`), and then pulled by the dashboard from the synced exports share so Listen chips stream immediately.

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

## Dashboard Notes (Postgres-only)
- Dashboard reads from Postgres only; it does not scan JSON or accept upload endpoints.
- Ensure at least one summary variant has non-null HTML so a card appears.
- `language` on `content` is used for language filtering.

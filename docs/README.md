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
  1. From the repo root, run `docker build -t ytv2-nas:latest .`
  2. In Portainer: **Containers → youtube-summarizer-bot → Duplicate/Edit → Replace existing container**, set Image to `ytv2-nas:latest`, confirm env vars, then deploy.
  3. After the new container starts, run `python tools/test_postgres_connect.py` from the console to confirm psycopg is available.
- Portainer: Use **Recreate** with _Re-pull image_ **off** to pick up code changes from the bind-mounted repo.
- After updating `.env.nas` (e.g., new Reddit refresh token), recreate the container so env vars reload.
- Helper scripts:
  - `tools/setup_postgres_schema.py` — schema/grants bootstrapping
  - `tools/debug_audio_variant.py` — inspect `content.has_audio` and `<audio>` HTML served from Postgres exports
  - `tools/test_reddit_connection.py`, `tools/debug_reddit_env.py` — Reddit diagnostics

## Dashboard Notes (Postgres-only)
- Dashboard reads from Postgres only; it does not scan JSON or accept upload endpoints.
- Ensure at least one summary variant has non-null HTML so a card appears.
- `language` on `content` is used for language filtering.

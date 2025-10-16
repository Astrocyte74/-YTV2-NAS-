# YTV2-NAS Operational Notes

## Telegram Bot Content Flow
- Unified pipeline supports both YouTube videos and Reddit threads.
- Telegram stores the active item context (`source`, `url`, `content_id`, etc.) so all summary types reuse the same keyboard.
- Summaries are exported locally (JSON optional) and written directly to Postgres via UPSERTs; ledger keys use universal IDs (`yt:<id>`, `reddit:<id>`).
- Audio variants remain available for Reddit because the summarizer operates on the combined thread text.

## Reddit Integration
- Credentials required: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (blank for installed app), `REDDIT_REFRESH_TOKEN`, `REDDIT_USER_AGENT`.
- Fetcher handles canonical URLs, `redd.it/<id>`, and the short `/r/<sub>/s/<token>` share links.
- JSON reports include `content_source="reddit"`, subreddit metadata, and comment snippets for context.

## Deployment & Container Tips
- PRAW is installed via `requirements.txt`; rebuild the NAS image (`docker build -t ytv2-with-ffmpeg .`) when dependencies change.
- Portainer: Use **Recreate** with _Re-pull image_ **off** to pick up code changes from the bind-mounted repo.
- After updating `.env.nas` (e.g., new Reddit refresh token), recreate the container so env vars reload.
- Helper scripts (`tools/test_reddit_connection.py`, `tools/debug_reddit_env.py`) can be copied into the container for quick verification.

## Dashboard Notes (Postgres-only)
- Dashboard reads from Postgres only; it does not scan JSON or accept upload endpoints.
- Ensure at least one summary variant has non-null HTML so a card appears.
- `language` on `content` is used for language filtering.

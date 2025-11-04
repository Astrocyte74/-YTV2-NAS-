# Tools & Diagnostics

Utility scripts to assist with operations, testing, and migration tasks. Run these inside the NAS container unless otherwise noted.

## Core DB Utilities

- `setup_postgres_schema.py` – Creates/patches tables, view, indexes, and grants.
- `test_postgres_connect.py` – Connectivity and role permission smoke test.
- `test_upsert_content.py` – Inserts one `content` row and summary variants; verifies Listen eligibility.
- `delete_postgres_video.py` – Removes test rows from `content`/`summaries` by `video_id`.

## Backfill & Diagnostics

- `debug_audio_variant.py` – Prints `content.has_audio` and `<audio>` HTML for a given `video_id`.
- `list_audio_rows.py` – Lists summary rows matching a `video_id` substring (useful for debugging legacy prefixes).
- `strip_yt_prefix_in_summaries.py` – Cleans older `yt:`-prefixed `video_id` rows.
- `backfill_metadata.py`, `backfill_analysis.py`, `cleanup_reports.py` – Historical JSON maintenance scripts. Review source before use.
- `batch_fix_audio_urls.py` – Finds local MP3s and fixes rows with `has_audio=true` but missing `media.audio_url` by uploading and updating Postgres. Use `--limit`, `--dry-run`. Optional `--respect-storage` (aborts when `/api/health/storage` used_pct ≥ threshold; configurable via `--block-pct` or `DASHBOARD_STORAGE_BLOCK_PCT`, default 98).
- `scan_and_fix_from_exports.py` – Scans `/app/exports` for MP3s and fixes corresponding rows (even when `has_audio=false`). Use `--limit`, `--cap`, `--dry-run`. Optional `--respect-storage` and `--block-pct` as above.
- `cleanup_audio_variants_no_url.py` – Removes `summaries` rows where `variant='audio'` but `content` has no `media.audio_url` and `has_audio=false`.
- `cleanup_broken_audio_cards.py` – Audits for cards that claim audio but have no playable MP3 (HEAD 200). Can dry‑run or delete.

## Deprecated / Legacy

- `test_api_sync.py`, `test_audio_upload.py`, `force_render_refresh.py` – Older scripts; keep for reference.
- `analyze_json_data.py`, other ad-hoc scripts – Useful for forensic work but not part of the core ingest path.

Notes
- Modern uploads use `POST /api/upload-audio` (primary) and `POST /api/upload-image` with either `Authorization: Bearer $SYNC_SECRET` or `X-INGEST-TOKEN: $INGEST_TOKEN`. `POST /ingest/audio` remains as a fallback.
- After upload success, verify with HEAD on `/exports/audio/<filename>.mp3?v=<audio_version>` and update Postgres fields (`has_audio`, `media.audio_url`, `media_metadata.mp3_duration_seconds`, `audio_version`).
- Admin/Health:
  - `tools/dashboard_health.py` prints `/api/version` and (if gated) `/api/health/storage`; exits non‑zero at critical.
  - Set `DASHBOARD_DEBUG_TOKEN` on NAS to enable storage probes (matches dashboard `DEBUG_TOKEN`).

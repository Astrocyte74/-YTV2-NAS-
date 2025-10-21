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

## Deprecated / Legacy

- `test_api_sync.py`, `test_audio_upload.py`, `force_render_refresh.py` – Relics from the HTTP ingest era. Keep for reference; do not run against the Postgres-only stack.
- `analyze_json_data.py`, other ad-hoc scripts – Useful for forensic work but not part of the live ingest path.

> Dashboard is Postgres-only. Avoid HTTP ingest endpoints; rely on `PostgresWriter` + the tools above.

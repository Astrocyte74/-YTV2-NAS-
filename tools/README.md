# Tools & Diagnostics

Utility scripts to assist with operations, testing, and migration tasks. Run these from the repository root with the virtual environment active unless otherwise noted.

- DEPRECATED: `test_api_sync.py` – Render ingest API is removed; dashboard is Postgres-only. Keep for reference; do not use in production.
- DEPRECATED: `test_audio_upload.py` – Audio upload endpoints are gone. Use audio variants written to Postgres instead.
- `test_ffprobe.py` – Verifies that `ffprobe` can extract duration metadata from generated audio files.
- `force_render_refresh.py` – Legacy; dashboard no longer scans JSON or provides refresh endpoints.
- `analyze_json_data.py`, `backfill_*`, `cleanup_reports.py`, etc. – Historical scripts used for data backfills or maintenance. Consult the source before running.

Planned replacements (DB-first):
- `tools/test_postgres_connect.py` – Connectivity and role permission smoke test.
- `tools/test_upsert_content.py` – Inserts one `content` row and a couple of summary variants.

> Note: The dashboard is Postgres-only. Write rows directly to the database; do not use HTTP ingest endpoints.

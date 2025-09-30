# Tools & Diagnostics

Utility scripts to assist with operations, testing, and migration tasks. Run these from the repository root with the virtual environment active unless otherwise noted.

- `test_api_sync.py` – Smoke test for the Render ingest API. Confirms connectivity and end-to-end content sync via REST rather than the legacy SQLite flow.
- `test_audio_upload.py` – Uploads an existing MP3 in `exports/` to the Render dashboard for a given `yt:<video_id>` content ID.
- `test_ffprobe.py` – Verifies that `ffprobe` can extract duration metadata from generated audio files.
- `force_render_refresh.py` – (Legacy) triggers a full re-sync of content using the API wrapper; primarily useful for forcing Render to pick up schema changes.
- `analyze_json_data.py`, `backfill_*`, `cleanup_reports.py`, etc. – Historical scripts used for data backfills or maintenance. Consult the source before running.

Most scripts prepend the project root and `modules/` directory to `sys.path` automatically, so they can be executed directly, e.g.:

```bash
python tools/test_api_sync.py
python tools/test_audio_upload.py yt:TuEpUrQCOkk
```

> **Note:** The old SQLite-only utilities have been removed. These helpers all interact with the Postgres/Render APIs.

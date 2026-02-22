# YTV2 API Migration Guide (Deprecated)

This file is retained only for historical reference. The NAS now writes directly to Postgres and uploads audio via `/api/upload-audio`. The HTTP ingest phase described here has been removed.

## Current Workflow

- Use `modules/postgres_writer.PostgresWriter` for `upload_content` and `upload_audio`.
- Run `python tools/setup_postgres_schema.py` once to create tables, view, indexes, and grants.
- Verify with `python tools/test_postgres_connect.py` and `python tools/test_upsert_content.py`.
- Audio: NAS pushes MP3s to Postgres (`content.has_audio=true`) and to Render (`/exports/audio/<file>.mp3`).

Refer to these living docs instead:

- [`POSTGRES_UPSERT_GUIDE.md`](POSTGRES_UPSERT_GUIDE.md) — schema, UPSERT patterns, role grants.
- [`README.md`](README.md#first-run-checklist) — bootstrap + smoke tests.
- [`docs/README.md`](docs/README.md) — operational notes for NAS vs Render.

The legacy HTTP ingest instructions have been archived in `archive/API_MIGRATION_GUIDE_legacy.md` should they ever be needed.

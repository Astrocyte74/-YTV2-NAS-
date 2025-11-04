# NAS ↔ Render Integration (Postgres + Uploads)

This document summarizes how the NAS syncs summaries and audio to the Render‑hosted dashboard and what env vars are required.

Overview
- Content writes: direct PostgreSQL upsert from NAS
- Audio/Image uploads: authenticated multipart uploads to the dashboard
- Read APIs: `/api/reports`, `/api/filters`, `/<id>.json` enriched with audio variants

Environment Variables
- NAS:
  - `DATABASE_URL` – Postgres DSN used by the NAS for direct content upserts
  - `RENDER_DASHBOARD_URL` – e.g., https://ytv2-dashboard-postgres.onrender.com
  - `INGEST_TOKEN` – copy from Render service env; used for private ingest
  - `SYNC_SECRET` – optional bearer token; either header works for uploads
  - Optional: `DUAL_SYNC`, `POSTGRES_ONLY` (we run `POSTGRES_ONLY=true`)
  - Optional: `AUDIO_PUBLIC_BASE` – not required; use HTTP uploads instead
- Render (dashboard):
  - `DATABASE_URL_POSTGRES_NEW` – Postgres DSN
  - `INGEST_TOKEN` – shared secret for NAS ingest

Endpoints (Render)
- JSON read:
  - `GET /api/reports` — list view; enriched `summary_variants`
  - `GET /<video_id>.json` — single view; enriched `summary_variants`, `has_audio`
- Uploads (multipart):
  - `POST /api/upload-audio` — primary audio upload (accepts `Authorization: Bearer` or `X-INGEST-TOKEN`)
  - `POST /api/upload-image` — summary image upload (same auth)
  - Fallback: `POST /ingest/audio` — legacy ingest
- Health checks:
  - `GET /health/ingest` → `{status:'ok', token_set:true, pg_dsn_set:true}`
  - `GET /api/reports?size=1` → sanity read

Audio Upload Strategy
- Preferred: Direct Postgres for metadata + `POST /api/upload-audio` for the MP3.
- Why: The dashboard serves audio from its filesystem; uploads ensure the file exists and then JSON enrichment pulls `audio_url`/`duration` from `content`.
- Coordinator behavior (NAS):
  1) Upload via `/api/upload-audio` with either `Authorization: Bearer $SYNC_SECRET` or `X-INGEST-TOKEN: $INGEST_TOKEN`.
  2) On success, update Postgres: `has_audio=true`, `media.audio_url`, `media_metadata.mp3_duration_seconds`, `audio_version`.
  3) Fallback: `/ingest/audio` is available if needed.

Authoritative fields written by NAS (after 200 OK upload)
- `content.has_audio = true`
- `content.media.audio_url = "/exports/audio/<filename>.mp3"` (root‑relative)
- `content.media_metadata.mp3_duration_seconds = <int>` (ffprobe)
- Optional: `content.audio_version = <unix_ts>` (dashboard appends `?v=`)

JSON semantics
- Prefer omission over null when a field is unknown/unavailable (e.g., omit `audio_url` and `mp3_duration_seconds` if upload fails).

Typical Logs
- Direct PG content upsert:
  - `[SYNC] target=postgres ... op=report status=ok upserted=True`
- Skipping DB audio variant (expected):
  - `AUDIO_PUBLIC_BASE not set; skipping DB audio variant (HTTP ingest fallback may upload the MP3).`
- HTTP ingest fallback for MP3:
  - `[SYNC] target=postgres ... op=audio status=fallback_http_ingest`
  - `✅ PostgreSQL audio uploaded: <video_id> -> /exports/audio/<sanitized>.mp3`
  - `[SYNC] ... op=audio status=ok (http)`

Troubleshooting (MP3 not playable)
1) Verify health: `curl -sS "$RENDER_DASHBOARD_URL/health/ingest" | jq`
   - Expect `token_set:true`, `pg_dsn_set:true`.
2) Check the returned URL: `curl -I "$RENDER_DASHBOARD_URL/exports/audio/<filename>.mp3?v=<audio_version>"`
   - Expect `200` with non‑zero `Content-Length`. If `404` or size 0, re‑upload.
3) Look for coordinator logs:
   - `status=fallback_http_ingest` followed by `status=ok (http)` indicates success.
4) Disk full on Render produces 500s and zero‑byte artifacts; increase `/app/data` or clean old files. Server writes atomically to avoid partial files.

Notes
- Upload responses include `public_url`/`relative_path` and `size`; prefer server‑returned paths.
- The server sanitizes filenames and keeps DB `video_id` stable.
- Send consistent `video_id` between content and audio calls.

## Admin & Health (Optional)

Admin endpoints (gated with `DEBUG_TOKEN` on the dashboard; use `DASHBOARD_DEBUG_TOKEN` on NAS):
- `GET /api/version` (no auth) — deploy metadata
- `GET /api/health/storage` (`Authorization: Bearer <token>`) — `{ total_bytes, used_bytes, free_bytes, used_pct }`; returns 503 at ≥98% to signal critical
- `GET /api/debug/content?video_id=<id>` (Bearer token) — raw `content` row for ops

NAS usage:
- `/status` shows version + storage health when `DASHBOARD_DEBUG_TOKEN` is set
- Optional pre‑upload backoff when `used_pct ≥ DASHBOARD_STORAGE_BLOCK_PCT` (default 98%)
- Early chat gate: warn at ≥90/≥95% and block new processing at ≥99%

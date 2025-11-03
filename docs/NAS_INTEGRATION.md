# NAS ↔ Render Integration (Postgres + HTTP Ingest)

This document summarizes how the NAS syncs summaries and audio to the Render‑hosted dashboard and what env vars are required.

Overview
- Content writes: direct PostgreSQL upsert from NAS
- Audio uploads: HTTP ingest to Render (preferred) with token auth
- Read APIs for UI remain unchanged (`/api/reports`, `/api/filters`)

Environment Variables
- NAS:
  - `DATABASE_URL` – Postgres DSN used by the NAS for direct content upserts
  - `RENDER_DASHBOARD_URL` – e.g., https://ytv2-dashboard-postgres.onrender.com
  - `INGEST_TOKEN` – copy from Render service env; used for private ingest
  - Optional: `DUAL_SYNC`, `POSTGRES_ONLY` (we run `POSTGRES_ONLY=true`)
  - Optional: `AUDIO_PUBLIC_BASE` – not required when using HTTP ingest
- Render (dashboard):
  - `DATABASE_URL_POSTGRES_NEW` – Postgres DSN
  - `INGEST_TOKEN` – shared secret for NAS ingest

Endpoints (Render)
- `POST /ingest/report` (JSON): upserts content by `video_id`
- `POST /ingest/audio` (multipart): saves MP3 and flips `has_audio=true`
- Health checks:
  - `GET /health/ingest` → `{status:'ok', token_set:true, pg_dsn_set:true}`
  - `GET /api/reports?size=1` → sanity read

Audio Upload Strategy
- Preferred: Keep direct Postgres for content metadata, and upload MP3s via HTTP ingest.
- Why: The dashboard serves audio from its filesystem; HTTP ingest ensures the file exists on Render and updates DB flags.
- Coordinator behavior (NAS):
  1) Try direct Postgres audio variant insert only if `AUDIO_PUBLIC_BASE` is set.
  2) If not set or insert returns None, automatically fall back to HTTP ingest using `RENDER_DASHBOARD_URL` + `INGEST_TOKEN`.
- Result: No `AUDIO_PUBLIC_BASE` required; audio reliably appears on the dashboard.

Authoritative fields written by NAS (after 200 OK upload)
- `media.has_audio = true`
- `media.audio_url = /exports/by_video/<videoId>.mp3` (Reddit legacy: `/exports/audio/reddit<file_stem>.mp3`)
- `media_metadata.mp3_duration_seconds = <int>` (ffprobe)
- Optional: `audio_version = <unix_ts>` (dashboard may append `?v=`)

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
2) Check the by‑video URL: `curl -I "$RENDER_DASHBOARD_URL/exports/by_video/<video_id>.mp3"`
   - Expect `200`. If `404`, the MP3 didn’t upload; confirm `INGEST_TOKEN` and `RENDER_DASHBOARD_URL` on NAS.
3) Look for fallback logs:
   - `status=fallback_http_ingest` followed by `status=ok (http)` indicates success.
4) If you must use direct PG for audio URLs, set `AUDIO_PUBLIC_BASE` to a real, public base that serves your NAS files (not recommended).

Notes
- The server sanitizes filenames but updates DB using the original `video_id` (e.g., `web:`/`reddit:` prefixes are preserved).
- Send `video_id` consistently between content and audio calls.

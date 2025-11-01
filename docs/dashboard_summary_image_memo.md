# Dashboard Memo: Summary Illustration Support

Team—thanks again for the quick work on `/api/upload-image`. Here’s the up-to-date picture so UI and API stay in sync.

## Current behaviour
- NAS now renders a 384×384 PNG for every summary (tech / gospel templates for now).
- PNG lands locally at `exports/images/<slug>_<timestamp>_<template><size>.png` and the hub seed + prompt are saved in the payload.
- During dual-sync we POST the PNG to `POST /api/upload-image`; the dashboard serves the mirrored file from `/exports/images/<filename>.png` (same static path as audio).
- Telegram replies with the illustration immediately after the summary message so reviewers see the art in-line.

## Environment & auth usage (after our audit)
- `RENDER_DASHBOARD_URL` – canonical base. NAS also backfills `RENDER_API_URL` when it’s missing. Props if we migrate everything to the `RENDER_DASHBOARD_URL` name.
- `INGEST_TOKEN` – still used by the legacy `/ingest/audio` path and now accepted by `/api/upload-image` (via the `X-INGEST-TOKEN` header).
- `SYNC_SECRET` – bearer token for `/api/upload-image`. NAS prefers this when set, otherwise falls back to `INGEST_TOKEN`.
- `AUDIO_PUBLIC_BASE` – optional; we leave it blank because HTTP ingest uploads already expose `/exports/…`.
- `POSTGRES_DASHBOARD_URL` / `DASHBOARD_URL` – older aliases still referenced in a few places (Telegram dashboard buttons, summary headers). They fall back to `RENDER_DASHBOARD_URL` when unset.

## Data model
- Postgres column: `content.summary_image_url` (nullable) – points at the public PNG.
- Optional future columns: `summary_image_seed` / `summary_image_style` if we decide they’re useful.
- JSON reports include `summary.summary_image` metadata (prompt, seed, template) and `summary_image_url` for dashboards/exports.

## Open questions for the dashboard team
1. **Token consolidation** – should we keep both `SYNC_SECRET` and `INGEST_TOKEN`, or can we standardise on a single token (e.g., ingest only)? Once you pick, we’ll adjust the NAS client and docs.
2. **Base URL naming** – okay to treat `RENDER_DASHBOARD_URL` as the only required variable and deprecate `RENDER_API_URL` / `POSTGRES_DASHBOARD_URL`, or do you rely on those names elsewhere?
3. **Long-term hooks** – do you want seed/style metadata exposed in the API for re-render or debugging, or is `summary_image_url` sufficient for now?
4. **Backfill** – if you plan to backfill images server-side, let us know the timeline so we can monitor NAS load and confirm the UI handles more image-bearing cards.

Once we have your answers we’ll:
- simplify the auth logic in the NAS client (e.g., drop the bearer path if not needed),
- clean up the doc references to legacy variables, and
- align the dashboard buttons with the canonical base URL.

Thanks again—really happy with how the pipeline is shaping up. Shout if you see anything off or want extra metadata for the UI.

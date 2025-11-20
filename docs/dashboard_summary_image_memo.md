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

## Resetting the Image Pipeline

The dashboard card data now lives in **Postgres** (ytv2-dashboard-postgres) while the UI files live in the separate `YTV2-Dashboard` repo that Render deploys. Keep these boundaries in mind when fixing or rerunning image jobs:

- Updating NAS scripts (this repo) does **not** redeploy Render. If you need a helper script (e.g., the cleanup utilities below) inside the dashboard container, either copy it manually or land the same file in the `YTV2-Dashboard` repo before redeploying there.
- Never “requeue from local reports.” The only source of truth for determining what needs art is Postgres. Scanning `/app/data/reports` or other local JSON drops will regenerate cards that no longer exist. Always drive requeues with SQL (missing `summary_image_url`) or with the helper scripts below.

### Cleanup helpers (NAS + Render)

Two small scripts now live under `scripts/` to prevent manual SQL mistakes:

1. `scripts/clear_summary_images.py`  
   - Filters `content` rows via `--after`, repeated `--contains` tokens, and/or `--ids-file`.  
   - Default mode is a dry run (prints each row). Re-run with `--apply` to null both `summary_image_url` and `analysis_json.summary_image_ai2_url`.  
   - Example (clean up the runaway 2025‑11‑17 batch):  
     ```bash
     python3 scripts/clear_summary_images.py \
       --after 2025-11-17T00:00:00 \
       --contains 20251117 \
       --apply
     ```
2. `scripts/delete_orphan_images.py`  
   - Compares every PNG under `/app/exports/images` (configurable via `EXPORTS_IMAGES_DIR`) to the filenames referenced in Postgres.  
   - Dry-run first (default), then add `--delete` to remove the orphaned files.

Copy the same files into the `YTV2-Dashboard` repo (or cat them into `/app/scripts` on the Render host) before running them there.

### Full pipeline reset (rare)

Only do a full wipe when you intentionally want to regenerate *all* summary art (new prompts/templates):

1. **Remove existing PNGs on the NAS**
   ```bash
   docker exec -i youtube-summarizer-bot rm -rf /app/exports/images/*
   ```
2. **Clear image metadata in Postgres** (reuse `scripts/clear_summary_images.py --apply --after 1900-01-01` if preferable) so every row becomes a fresh target:
   ```bash
   docker exec -i youtube-summarizer-bot python3 -c "import os, psycopg\nurl = os.getenv('DATABASE_URL_POSTGRES_NEW') or os.getenv('DATABASE_URL')\nassert url, 'missing DB URL'\nreset_sql = '''\nUPDATE content\n   SET summary_image_url = NULL,\n       analysis_json = COALESCE(analysis_json, '{}'::jsonb)\n                        - 'summary_image_prompt'\n                        - 'summary_image_prompt_last_used'\n                        - 'summary_image_selected_url'\n                        || jsonb_build_object('summary_image_variants', '[]'::jsonb)\n'''\nwith psycopg.connect(url) as conn:\n    with conn.cursor() as cur:\n        cur.execute(reset_sql)\n        conn.commit()\n"
   ```
3. **Re-run the image backfill** in manageable batches (SQL-driven list or `/app/tools/backfill_summary_images.py --limit 25`).  
4. **Monitor the drain worker** (`drain_image_queue`) and verify Render uploads succeed (`/api/upload-image` 200s, files visible under `/exports/images`).  

Optional: if you only want to regenerate selected video IDs, feed them to `clear_summary_images.py --ids-file ids.txt --apply` before backfilling that specific list.

### Previewing a backfill

`tools/backfill_summary_images.py` now supports `--plan-only` to list exactly which cards would be touched without hitting the Draw Things hub:

```bash
python3 tools/backfill_summary_images.py --limit 20 --plan-only
```

Use this before running a real batch so you can sanity‑check the targets (combine with `--mode ai2`, `--only-missing-thumbnail`, or `--video-id` filters as needed). When you’re satisfied, drop `--plan-only` and run the same command to actually generate/upload the art. The existing `--dry-run` flag is still available if you ever want to exercise the image hub without writing to Postgres.

Prefer a guided flow? Run the interactive helper:

```bash
python3 tools/backfill_images_cli.py
```

It will prompt for limit, mode, plan-only/dry-run, and optional video IDs, then invoke the same backfill script with your selections.

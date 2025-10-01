# Future Considerations

A quick backlog of backend enhancements to explore once the current Postgres-only pipeline is settled:

- **Structured Monitoring & Alerts**
  - Add a lightweight metrics endpoint (Prometheus-style JSON/SSE) that exposes ingest latency, success/failure counts, TTS errors, SSE clients, etc.
  - Hook up a simple alert (Slack/webhook) to warn when ingest returns 4xx/5xx or when the queue stalls.

- **Report Reprocess API**
  - Provide a secure `/api/reprocess` endpoint (and Telegram command) that re-runs summaries/TTS for an existing video without re-submitting links.
  - Integrate with the ledger so operators can regenerate stale variants or apply new prompt tweaks.

- **Realtime Event Expansion**
  - Broadcast additional SSE events (`audio-ready`, `report-updated`, `report-deleted`) so the dashboard can update cards without polling.
  - Batch/merge events to prevent UI thrash when multiple variants arrive at once.

- **Metadata Enrichment**
  - Optional YouTube Data API integration to fetch reliable duration/channel info when yt-dlp fails.
  - Cache watch-page metadata to reduce repeated scraping and rate limits.

- **Summarizer Guardrails**
  - Track summary language vs. transcript language; auto re-run with alternate prompts/models when the AI returns empty or short summaries.
  - Enforce per-variant word counts and add QA hooks for new summary types.

- **Backfill & Housekeeping**
  - Finalize the metadata backfill script to populate new engagement/technical fields for legacy JSON.
  - Provide reindex/repair utilities for Postgres dashboards after major prompt/model changes.

These are intentionally scoped so they can be picked up incrementally without destabilising the existing workflow.

## ✅ Progress Checkpoint (2025-09-09)
- Added `modules/metrics.py` and integrated ingest/audio/TTS counters so `/api/metrics` exposes live stats.
- Introduced `modules/event_stream.py` powering the `/api/report-events` SSE endpoint with safe client handling.
- DualSync now emits `report-synced`/`audio-synced` plus failure events for dashboard listeners.
- POST `/api/reprocess` (and the `YouTubeTelegramBot.reprocess_video` flow) now headlessly reruns summaries and TTS, updating ledger entries and sync targets.
- Reprocess jobs increment new `reprocess_*` metrics and broadcast `reprocess-*` events for observability.
- `/api/reprocess` now enforces an `X-Reprocess-Token` header that must match `REPROCESS_AUTH_TOKEN` (set in container env).

### Next refinements
- Guard `/api/reprocess` with an auth token/shared secret before exposing beyond trusted LAN tools.
- Capture latency timings (ingest duration, TTS walltime) so the metrics snapshot can warn on slowdowns.
- Build a lightweight dashboard/CLI widget that consumes `/api/metrics` + SSE for NAS-friendly monitoring.

## 🧪 Verification Log (2025-09-30)
- Restarted the NAS container (Portainer redeploy) to load updated modules.
- `curl http://192.168.4.54:6452/api/metrics` ➜ counters returned with default zeros.
- `curl -N http://192.168.4.54:6452/api/report-events` ➜ SSE connection established (keep-alive heartbeats).
- `curl -X POST /api/reprocess` with `{video_id:"hvLV6xDGll0", summary_types:["comprehensive"], regenerate_audio:false}` ➜ SSE emitted `reprocess-scheduled`, `reprocess-requested`, `report-synced`, `reprocess-complete`.
- Metrics after the run: `ingest_success=1`, `reprocess_requested=1`, `reprocess_success=1`, `sse_events_broadcast=4`, `last_ingest_video="yt:hvLV6xDGll0"`.
- Re-queued container with `REPROCESS_AUTH_TOKEN` and verified that requests without `X-Reprocess-Token` receive HTTP 403.

## 🔗 Front-End Handoff Notes
- Subscribe to `/api/report-events` and handle:
  - `report-synced`, `audio-synced`
  - `report-sync-failed`, `report-sync-error`, `audio-sync-failed`, `audio-sync-error`
  - `reprocess-scheduled`, `reprocess-requested`, `reprocess-complete`, `reprocess-error`
- Poll/cache `/api/metrics` for dashboard widgets (ingest/audio/TTS/reprocess counters, `sse_clients_current`, last-ingest metadata).
- UX ideas: tiny status pill for last ingest, toast when `reprocess-complete`, badge when failures arrive, mini chart for per-day counts.
- Auth now active: include `X-Reprocess-Token: <secret>` in every `/api/reprocess` call (secret lives in container env `REPROCESS_AUTH_TOKEN`).

## 🎛️ Variant Reference (Backend Vocabulary)
- Text summaries: `comprehensive`, `bullet-points`, `key-insights`
- Audio variants: `audio`, `audio-fr`, `audio-es`
- Language-learning modes: append `:beginner`, `:intermediate`, or `:advanced` to multilingual audio variants (e.g., `audio-fr:beginner`)
- `/api/reprocess` tips:
  - Omit `summary_types` to re-run every variant already present for that video.
  - Provide an explicit list (e.g., `["audio","key-insights"]`) to target specific outputs.
  - Set `regenerate_audio:true` when you need fresh TTS; `false` reuses the last MP3 path if available.

### Item Status Snapshot
- **Structured Monitoring & Alerts**: Metrics endpoint live; alerting/webhook work still pending.
- **Report Reprocess API**: Implemented, auth-guarded, verified end-to-end.
- **Realtime Event Expansion**: Core ingest/audio/reprocess events broadcasting; batching/extra event types remain optional future work.

• I'm outlining and implementing the reprocess_video method for YouTubeTelegramBot, including helpers for extracting video URLs, generating TTS audio, updating
  ledger entries, and syncing reports. I’m also planning a new HTTP handler function handle_reprocess_request to schedule reprocess jobs, deduplicate tasks,
  respond properly, and integrate metrics for success or failure. This involves carefully adapting existing ledger, dual-sync, and summarizer logic while ensuring
  good fallback behaviors and error handling. I’ll add static helpers for video ID extraction and JSON report-based URL resolution, plus metrics tracking around
  audio processing outcomes. The changes touch both telegram_bot.py and modules/telegram_handler.py.

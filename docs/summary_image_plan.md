# Summary Graphic Generation Plan

## Why
- Enhance summaries with bespoke visuals that reflect our AI-written take instead of relying solely on source thumbnails.
- Provide richer assets for dashboards, Telegram shares, and future social repost tooling (two-image layout: source + summary graphic).
- Reuse the Draw Things hub we already operate, keeping control of style, latency, and cost.

## Scope
- **Sources:** Any summary we already generate (YouTube, Reddit, web articles). Start with long-form video summaries.
- **Output:** One additional image per summary (`summary_image_url`) sized for card-style usage (~384×384).
- **Status:** Planning & validation only—no production wiring yet.

## Proposed Pipeline
1. **Trigger point**  
   After we produce the final summary payload (`youtube_summarizer.process_text_content` / `process_youtube_video`) but before we upsert to Postgres.

2. **Prompt assembly**  
   - Extract key fields: title, channel/site, top bullet(s) or TL;DR.  
   - Feed into a lightweight template → narrative scene description.  
   - Optionally pass through existing prompt enhancers (local vs. cloud) with a “summary card” style hint.

3. **Image generation**  
   - Reuse `modules/services/draw_service.generate_image()`.  
   - Target Flux preset (Balanced or Detail) at 384×384; capture the hub seed for reproducibility.  
   - 1 retry with fallback settings (same logic as Telegram bot).

4. **Storage & mirroring**  
   - Download the generated PNG from the hub and persist alongside exported media.  
   - Path: `exports/images/<slug>_<timestamp>_<template><size>.png` under the NAS repo.  
   - During dual-sync, POST the file to the dashboard (`POST /api/upload-image`) using the same auth token as audio uploads so the Render instance serves `/exports/images/<filename>.png`.  
   - Keep the seed + source URL in metadata for re-rendering/debug.

5. **Persistence**  
   - Extend Postgres schema (`content.summary_image_url`, optionally `summary_image_seed`).  
   - Update `PostgresWriter._to_db_payload` to include the new column and adjust SQLite mirror when enabled.
   - JSON reports carry the same field so archived copies remain self-contained.

6. **Surfacing**  
   - Dashboard: display side-by-side with source thumbnail, fallback gracefully if generation failed.  
   - Telegram: bot now replies with the illustration immediately after a summary finishes, so reviewers can see the card styling inline.

## Storage & Distribution
- **Thumbnails today:** We reference remote URLs provided by YouTube/Reddit/web metadata. No NAS storage.  
- **Generated art:** Needs local persistence; the `exports/images/` tree (mirrored to Render via `/api/upload-image`) is the canonical location.  
- **Serving path:** dashboard serves images at `/exports/images/<filename>.png`, matching the MP3 static route.

## Operational Considerations
- Generation adds 5–10 s per summary; gate behind a feature flag (`SUMMARY_IMAGE_ENABLED`).  
- Handle hub outages gracefully (skip and continue).  
- Capture seeds for reproducibility; allow manual regenerate hooks later.  
- Respect content safety: skip when summary is marked sensitive / NSFW.

## Open Questions
- Should we run generation synchronously during summary creation or queue it as a follow-up job?  
- How do we surface errors or manual overrides (regenerate with edited prompt)?  
- Do we want style families (e.g., Flux narrative vs. minimalist infographic) per content type?

## Next Steps
1. Validate a universal prompt template on existing summaries (current task).  
2. Decide on storage path + schema migration details.  
3. Prototype a helper module (`summary_image_service`) with feature flag and metrics.  
4. Add CLI/telemetry to monitor generation success rates.  
5. Roll out behind config and update Dashboard to display the new asset.

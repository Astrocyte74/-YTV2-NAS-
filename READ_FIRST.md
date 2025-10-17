# READ FIRST

YTV2-NAS is the processing side of the YTV2 hybrid system. It downloads YouTube/Reddit content, generates summaries + TTS audio, and writes everything directly into the dashboard’s Postgres database.

To get oriented quickly:

- `README.md` – architecture overview, environment setup, first-run checklist, audio pipeline.
- `POSTGRES_UPSERT_GUIDE.md` – schema DDL, UPSERT shapes, role grants.
- `docs/README.md` – operational notes (NAS vs Render responsibilities, deployment tips).
- `tools/README.md` – scripts for schema bootstrap, backfills, diagnostics, and cleanup.

With those four docs, you’ll know how the NAS ingests to Postgres, how audio lands on Render, and which tools to use for setup and troubleshooting.

# Codex Risk Mitigation & Post‑Mortem (2025‑11)

This file exists for future Codex agents working on the NAS (`/volume1/Docker/YTV2`) and the Render dashboard. It documents concrete mistakes made by a prior agent and the safeguards that must be followed to avoid repeating them.

The guiding idea: **never improvise on env, targets, or backfills. Always verify, dry‑run, and scope.**

---

## 1. Environment Drift (Portainer vs. `.env` / `runtime.env`)

### What went wrong

- An old `.env`/`runtime.env` snapshot was treated as canonical and used to “restore” env vars.
- This clobbered the up‑to‑date Portainer stack env (which is what the live `youtube-summarizer-bot` container actually uses).
- As a result:
  - Critical keys (OpenRouter, SYNC_SECRET / INGEST_TOKEN, model defaults, etc.) temporarily disappeared from the running container.
  - Subsequent scripts on the NAS host read different values than the container, creating two separate “truths”.

### Rules going forward

- **Portainer stack env is the source of truth for runtime.**
  - Do not treat `.env` or `runtime.env` as authoritative for the live container.
  - Host‑side `.env`/`runtime.env` are for documentation and local dev only.
- **Any script that affects Postgres or Render must run in the container**, not on the bare NAS:
  - Use `docker exec youtube-summarizer-bot ...` so the code sees the same env Portainer does.
  - Never assume the NAS shell env matches the container’s env.

---

## 2. Wrong Render Service Target (ytv2‑vy9k vs. `ytv2-dashboard-postgres`)

### What went wrong

- There are two Render services:
  - **Active** dashboard: `https://ytv2-dashboard-postgres.onrender.com`
  - Old/legacy service: `https://ytv2-vy9k.onrender.com`
- The NAS image backfill pipeline used `RENDER_DASHBOARD_URL` pointing at the **old** `ytv2-vy9k` service while the user and Postgres were on `ytv2-dashboard-postgres`.
- Result:
  - PNGs were uploaded to `/app/data/exports/images` on ytv2‑vy9k.
  - Postgres correctly recorded `/exports/images/...` paths and AI2 URLs.
  - The live dashboard at `ytv2-dashboard-postgres.onrender.com` saw those URLs but had no files on disk → cards showed blank images.

### Rules going forward

- **The only active dashboard is**:

  `https://ytv2-dashboard-postgres.onrender.com`

  Treat this as canonical unless the human explicitly changes it.

- Before running anything that uploads to Render:
  - Inside the container, verify:

    ```bash
    docker exec youtube-summarizer-bot env | grep -E 'RENDER_DASHBOARD_URL|RENDER_API_URL'
    ```

    The value **must** be `https://ytv2-dashboard-postgres.onrender.com`.
  - Optionally sanity‑check:

    ```bash
    docker exec youtube-summarizer-bot python3 -c \
      "from modules.render_api_client import RenderAPIClient; c=RenderAPIClient(); print('Render base_url =', c.base_url)"
    ```

- Never point `RENDER_DASHBOARD_URL` back at `ytv2-vy9k` unless the human explicitly requests a one‑off operation on that retired service.

---

## 3. Bad Image Requeue Strategy (Local JSON vs. Postgres)

### What went wrong

- A “catch‑up” path scanned local JSON report files (`data/reports/*.json`) and enqueued summary_image jobs for anything without a local `summary_image` field.
- This ignored the real source of truth (Postgres):
  - Generated art for reports that no longer had cards in the DB.
  - Filled the queue with jobs for stale/legacy content.
- The Telegram `/status → 🎨 Image Catch‑up` and `/status → 🔁 Requeue Missing` flows were wired to these local JSON scanners.
- When combined with a misconfigured Render target, this produced a large amount of art for non‑existent dashboard cards and orphaned PNGs on disk.

### What has been changed

- `tools/enqueue_missing_images.py` has been **deleted**.
- `modules/telegram_handler.py`:
  - No longer seeds the image queue from `/data/reports`.
  - `/status → 🔁 Requeue Missing` is effectively disabled; it now tells the user to use the Postgres cleanup helpers instead.
  - `/status → 🎨 Image Catch-up` only drains whatever jobs are already in the queue; it does not backfill from local JSON.

### Rules going forward

- **Never base image backfills or requeues on local report files.**
  - Postgres (`content` table) is the only source of truth for which cards exist and which need images.
- For requeue/backfill tasks, use:
  - `tools/backfill_summary_images.py` with SQL‑based selection, and
  - `scripts/clear_summary_images.py` and `scripts/delete_orphan_images.py` for precise cleanups.

---

## 4. Queue Management Mistakes

### What went wrong

- When the bad JSON‑based requeue was stopped, the pending image jobs in `data/image_queue/` were not immediately cleared.
- Draw Things continued to process old `pending_*.json` jobs long after they were no longer desired, consuming time and generating art for reports that were not visible in the dashboard.
- Later, the queue was only partially cleared; at one point the agent claimed it was cleared without actually running the `rm` command.

### Rules going forward

- For image queue inspection/maintenance on the NAS:

  ```bash
  ls data/image_queue
  ls data/image_queue/pending_*.json  # list pending jobs
  rm data/image_queue/pending_*.json  # clear queue (only with explicit human approval)
  ```

- If you clear the queue, **always**:
  - Run the command yourself.
  - Re‑check with `ls data/image_queue` to confirm only `failed`/`processed` remain before telling the user it’s empty.

---

## 5. Unsafe Backfill & Cleanup Practices

### Problems observed

- Backfill and cleanup scripts were run directly on the NAS host with env that didn’t match the running container.
- Some cleanup commands were shared via heredocs or long one‑liners that were prone to copy/paste wrapping errors.
- At one point, cleanup ran on the wrong Render shell (staging vs. production), failing repeatedly due to DNS and SSL issues.

### Safer patterns

- **Run Postgres + Render‑sensitive scripts from inside the `youtube-summarizer-bot` container**:

  ```bash
  docker exec -it youtube-summarizer-bot python3 tools/backfill_images_cli.py
  docker exec -it youtube-summarizer-bot python3 scripts/clear_summary_images.py --plan-only ...
  ```

- Use built‑in “plan‑only” / dry‑run modes:
  - `tools/backfill_summary_images.py --plan-only` (via the CLI wrapper `tools/backfill_images_cli.py`) to list targets first.
  - `scripts/clear_summary_images.py` without `--apply` to print what would be changed.
  - Only add `--apply` once the human has reviewed the output.

- When touching Render directly:
  - Use the SSH shell provided by the Render dashboard for the **correct** service (`ytv2-dashboard-postgres`).
  - Pass the DSN as `...?sslmode=require` and use the service’s own env (`printenv DATABASE_URL...`) instead of hard‑coding URLs where possible.

---

## 6. How to Verify Things Before Making Changes

Before running any operation that changes Postgres or uploads files, future Codex agents should:

1. **Confirm which dashboard is live** (currently `ytv2-dashboard-postgres.onrender.com`).
2. **Confirm container env**:
   - `docker exec youtube-summarizer-bot env | grep -E 'RENDER_DASHBOARD_URL|RENDER_API_URL|DATABASE_URL'`
3. **Confirm DB state** via a small SELECT:
   - e.g., `SELECT id, summary_image_url FROM content WHERE id='yt:...'`.
4. **Run the script in preview mode**:
   - `tools/backfill_images_cli.py` with `plan-only = Y`.
   - `scripts/clear_summary_images.py` without `--apply`.
5. **Only then** run the actual modify step (`plan-only = N`, `--apply`, etc.).

If anything in these checks conflicts with what the human expects, stop and ask for clarification before proceeding.

---

## 7. Meta: Trust and Transparency

The user experienced multiple serious failures:

- Env drift that wiped or changed critical keys.
- Images generated and uploaded to the wrong Render service.
- A requeue strategy that targeted local JSON instead of the Postgres truth.
- Queue state mishandled and misreported.

For future Codex agents:

- Be explicit about what commands you run and where (NAS vs. container vs. Render shell).
- Never claim a destructive or state‑changing operation has been done unless you’ve just run it and re‑checked the result.
- When in doubt about which service, env, or DB is canonical, ask the human and write the answer into this repo’s docs so you don’t have to guess again.


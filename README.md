# YTV2-NAS - YouTube Processing Engine

**The processing component** of the YTV2 hybrid architecture. This runs on your NAS and handles all YouTube video processing, AI summarization, and content generation.

## 🏗️ Architecture Overview

YTV2 uses a **hybrid architecture** with separated concerns:

- **🔧 NAS Component** (This project): YouTube processing + Telegram bot
- **🌐 Dashboard Component**: Web interface + audio playback (deployed to Render)

### How It Works (Postgres‑only)

1. **📱 Telegram Bot** receives YouTube URLs from users
2. **🤖 AI Processing** downloads, transcribes, and summarizes videos (Gemini Flash Lite by default)  
3. **📊 JSON Reports** generated with structured metadata and language-aware summaries
4. **🎵 Audio Export** produces TTS audio (single or multi-language) with vocabulary overlays
5. **🗄️ Direct DB Writes** upsert to Postgres (`content` + `summaries` backing `v_latest_summaries`)
6. **🌐 Web Access** users view summaries; cards appear when at least one variant has HTML

## ✨ Features

- **🤖 Telegram Bot Interface**: Send YouTube URLs for instant AI processing
- **🎯 AI-Powered Summarization**: Multiple summary types with sentiment analysis
- **🔄 Duplicate Prevention**: JSON ledger system prevents reprocessing videos
- **🎵 Audio Generation**: Multi-language TTS with vocabulary scaffolding (FR/ES variants)
- **📊 Structured Reports**: JSON + HTML summaries with language metadata and key topics
- **🗄️ Postgres‑only**: No SQLite; metadata is written via UPSERTs to Postgres
- **🧵 Reddit Thread Support**: Fetch saved Reddit submissions and summarize them alongside YouTube videos
- **📰 Web Article Support**: Layered extractor cleans arbitrary https links (Readability + Trafilatura fallbacks)
- **⚠️ Resilient Metadata**: Falls back to YouTube watch-page parsing when yt-dlp formats are blocked
- **⚙️ Multi-AI Support**: OpenRouter (Gemini Flash Lite), OpenAI, Anthropic
- **🔧 Docker Ready**: Easy NAS deployment via Portainer

## 🚀 Quick Setup

### Prerequisites

- **Docker** environment (Portainer recommended)
- **API Keys**: OpenAI, Anthropic, or OpenRouter
- **Telegram Bot**: Token from @BotFather
- **Postgres Access**: `DATABASE_URL` or `PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE/PGSSLMODE`

### Installation

1. **Clone this repository** to your NAS
2. **Copy environment template**:
   ```bash
   cp .env.template .env.nas
   ```

3. **Configure environment** (`.env.nas`):
   ```bash
   # Telegram Configuration
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_ADMIN_USER_ID=your_user_id_here
   
   # AI Provider (choose one)
   OPENAI_API_KEY=your_openai_key_here
   # OR
   ANTHROPIC_API_KEY=your_anthropic_key_here
   # OR  
   OPENROUTER_API_KEY=your_openrouter_key_here

   # Optional Reddit integration (non-YouTube sources)
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=
   REDDIT_REFRESH_TOKEN=your_reddit_refresh_token
   REDDIT_USER_AGENT="Summarizer by u/your_username"
   
   # Direct Postgres connection (dashboard DB)
   # Prefer a least-privileged role; SSL required on Render
   DATABASE_URL=postgresql://ytv2_ingest:password@host:5432/ytv2?sslmode=require

   # Optional: public base URL for audio files (to build audio variant links)
   # Example: https://your-host
   # Result URL becomes: ${AUDIO_PUBLIC_BASE}/exports/<filename>.mp3
   AUDIO_PUBLIC_BASE=

   # Feature flags
   POSTGRES_ONLY=true
   SQLITE_SYNC_ENABLED=false
   ```

4. **Deploy with Docker**:
   ```bash
   docker-compose up -d
   ```

### First Run Checklist

1. **Provision schema and grants**  
   `python tools/setup_postgres_schema.py`

2. **Verify connectivity and permissions**  
   `python tools/test_postgres_connect.py`

3. **Smoke-test inserts**  
   `python tools/test_upsert_content.py TEST1234567`  
   Clean up with `python tools/delete_postgres_video.py TEST1234567`.

4. **Confirm audio uploads**  
   Set `RENDER_DASHBOARD_URL` and `INGEST_TOKEN` on the NAS (match `INGEST_TOKEN` on Render).  
   Each TTS run uploads the MP3 to Render via `/ingest/audio` and flips `has_audio` in Postgres. See `docs/NAS_INTEGRATION.md`.

## 🔧 Configuration

### AI Provider Setup

Choose your preferred AI provider in `.env.nas`:

```bash
# OpenAI (Recommended)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_key_here

# Anthropic Claude
LLM_PROVIDER=anthropic  
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=your_key_here

# OpenRouter (Multiple Models)
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3-sonnet
OPENROUTER_API_KEY=your_key_here
```

### Dashboard Integration (Postgres + Uploads)

The dashboard reads from Postgres for content/variants and also accepts authenticated uploads for audio/images. The read path enriches response JSON so audio variants include `audio_url` and `duration` joined from the `content` row.

Write path (NAS side):
- Direct UPSERTs into Postgres (`content`, `summaries`) for metadata and text/html variants.
- Upload MP3s/images to the dashboard using:
  - `POST /api/upload-audio` (multipart) — primary
  - `POST /api/upload-image` (multipart)
  - Fallback: `POST /ingest/audio` is still available for compatibility
- Auth: either `Authorization: Bearer $SYNC_SECRET` or `X-INGEST-TOKEN: $INGEST_TOKEN`.

Read path (Dashboard JSON):
- `/<video_id>.json` and `/api/reports` include `summary_variants` with `{ kind:'audio', audio_url, duration }` when `content.media.audio_url` and `media_metadata.mp3_duration_seconds` are set.
- `has_audio` reflects the DB flag and/or presence of an enriched audio variant.

Requirements for cards to show:
- At least one summary variant must have non-null `html` (for text views) or an enriched audio variant.
- `language` on `content` is used for language filtering.

See `POSTGRES_UPSERT_GUIDE.md` and `docs/NAS_INTEGRATION.md` for DDL, role grants, upload, and health details.

### Backfill & Recovery Tools

| Purpose | Command |
| --- | --- |
| Bootstrap schema | `python tools/setup_postgres_schema.py` |
| Connectivity test | `python tools/test_postgres_connect.py` |
| Insert smoke test | `python tools/test_upsert_content.py <video_id>` |
| Inspect Postgres state | `python tools/debug_audio_variant.py <video_id>` |
| Clean legacy prefixes | `python tools/strip_yt_prefix_in_summaries.py` |
| Remove test data | `python tools/delete_postgres_video.py <video_id>` |

### Reddit Integration (Optional)

Enable Reddit thread ingestion by supplying OAuth credentials from a Reddit "installed app":

```bash
# Required for Reddit fetcher
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=         # leave blank for installed apps
REDDIT_REFRESH_TOKEN=your_refresh_token
REDDIT_USER_AGENT="Summarizer by u/your_username"
```

To generate these values:
- Create an "installed app" at https://www.reddit.com/prefs/apps
- Copy the `client_id` (the 14-character string under the app name)
- Leave the client secret empty
- Run the OAuth flow once to obtain a long-lived refresh token
- Use a descriptive user agent (Reddit recommends `app-name by u/<username>`)

## 📁 Project Structure

### Essential Files
```
YTV2-NAS/
├── telegram_bot.py          # Main Telegram bot
├── youtube_summarizer.py    # Video processing engine  
├── nas_sync.py              # Dashboard synchronization / Postgres ingest
├── export_utils.py          # Summary export utilities
├── llm_config.py            # AI model configuration
├── modules/                 # Processing utilities
│   ├── ledger.py         # Duplicate prevention
│   ├── render_probe.py   # Dashboard connectivity  
│   ├── services/         # Summary/TTS/Ollama service layer
│   └── telegram_handler.py # Bot interaction logic
├── tools/                  # Diagnostics & one-off scripts (see tools/README.md)
├── data/                   # Runtime reports/transcripts (ignored by Git)
├── exports/                # Generated audio files (ignored by Git)
└── config/                 # Configuration templates
```

### Archived Files
- `archive_nas/old_*` - Previous versions and unused utilities
- `archive/` - Old report backups

## 🐳 Docker Deployment

### Using Portainer (Recommended)

1. **Import Stack**: Use the provided `docker-compose.yml`
2. **Set Environment**: Upload your `.env.nas` file
3. **Deploy**: Start the stack

## 🔧 Bot Admin Commands

These Telegram commands are available to admin users (allowed IDs) to help operate and troubleshoot the NAS bot:

- `/status` (`/s`) — Overall health snapshot with inline buttons. Shows:
  - Summarizer and LLM config
  - Local LLM (hub/direct) provider, base, reachability, installed models
  - TTS hub favorites/engines, queue status, process uptime, Git SHA
  - Dashboard version and storage health (requires `DASHBOARD_DEBUG_TOKEN`)
  - Admin shortcuts: Diagnostics, Logs, Restart
- `/diag` — One-shot diagnostics inside the container:
  - Python/platform, yt-dlp/ffmpeg, disk usage, uptime
  - Local LLM reachability (hub/direct) and model count
  - Postgres connectivity if `DATABASE_URL` is set
- `/logs [N]` — Tail the last N lines from `bot.log` (default 80)
- `/restart` (`/r`) — Gracefully restarts the container (Compose restart policy restarts the bot); you will receive a confirmation message after it comes back online.

Notes
- Local LLM routing is handled by a unified client that prefers the hub when `TTSHUB_API_BASE` is set and falls back to direct Ollama when `OLLAMA_URL`/`OLLAMA_HOST` is set. Errors are normalized to enable clean cloud fallback.

### Storage Health & Gating (Dashboard)

- Admin endpoints on the dashboard (token‑gated via `DEBUG_TOKEN`):
  - `GET /api/version` (no auth) → deployment info
  - `GET /api/health/storage` (Bearer `DASHBOARD_DEBUG_TOKEN`) → `{ total_bytes, used_bytes, free_bytes, used_pct }` and recent files
  - `GET /api/debug/content?video_id=…` (Bearer token) → raw row preview for ops
- NAS probes and behavior:
  - The bot’s `/status` shows dashboard version + storage health when `DASHBOARD_DEBUG_TOKEN` is set.
  - Before uploads, the NAS client optionally probes `/api/health/storage` and backs off when `used_pct` ≥ `DASHBOARD_STORAGE_BLOCK_PCT` (default 98%).
  - Upload responses must include `size > 0`; zero‑size is retried.
  - Early gate in the message handler (when a user pastes a link):
    - Warn at ≥90% and ≥95% usage; block new processing at ≥99%.
    - This check is best‑effort and only runs when `DASHBOARD_DEBUG_TOKEN` and `RENDER_DASHBOARD_URL` are set.

Environment (admin probes):
- `DASHBOARD_DEBUG_TOKEN` – token matching dashboard `DEBUG_TOKEN` (enables probes)
- `DASHBOARD_STORAGE_BLOCK_PCT` – optional integer (default 98) for pre‑upload backoff threshold

## 🎨 Draw Things Prompting

- `/draw <prompt>` (`/d`) — open the Draw Things helper. The bot lets you:
  - Enhance the prompt with your local model (via Ollama) or cloud shortlist.
  - Generate Small (512²), Medium (768²), or Large (1024²) images through the hub convenience endpoint (`POST /telegram/draw`).
  - Receive the rendered image back in Telegram with metadata (steps, seed when available).
- Requires `TTSHUB_API_BASE` pointing at the Mac hub (WireGuard IP + port `7860`). The bot never calls Draw Things directly; it always uses the hub proxy.
- Configure `DRAW_MODELS` (comma-separated `name:group`) to control the model picker; defaults are `Flux.1 [schnell]:flux,HiDream I1 fast:general`. Models tagged `flux` surface Flux presets; all others use the general preset family.
- Preset data (`/telegram/presets`) is fetched from the hub so you can pick Flux / SDXL presets, style presets, and negative presets in-line; “Preset Size” uses the dimensions bundled with the current preset.
- While an image or enhancement is in flight the inline keyboard collapses to a disabled “Working…” state so users know to wait; status text beneath the prompt confirms success or failure.
- Container logs record each step (`draw: command…`, `draw: enhancing…`, `draw: generating…`, etc.) which is useful for troubleshooting.

4. **Monitor**: Check logs for successful startup

### Built‑in TTS Queue Worker

- The image runs both the Telegram bot and a background TTS queue watcher by default.
- No extra service is required; the entrypoint launches:
  - `python3 telegram_bot.py`
  - `python3 tools/drain_tts_queue.py --watch`
- Environment toggles:
  - `ENABLE_TTS_QUEUE_WORKER=1` (default) – set to `0` to disable
  - `TTS_QUEUE_INTERVAL=30` – poll interval in seconds
  - `POSTGRES_ONLY=true` – recommended for NAS; worker skips SQLite
- Manual drain (inside container):
  - `python3 tools/drain_tts_queue.py` (one‑shot)
  - `python3 tools/drain_tts_queue.py --watch --interval 15` (watch mode)

Note: After updating to the image with the new entrypoint, rebuild once and recreate the container (or re‑pull if using a registry).

### Manual Docker

```bash
# Build and run
docker-compose up -d

# View logs  
docker-compose logs -f

# Stop/restart
docker-compose down && docker-compose up -d
```

## 🔄 Usage Workflow

1. **Send YouTube or Reddit URL** to your Telegram bot
2. **Bot processes** video/thread (download/fetch → transcribe/aggregate → summarize)
3. **Optional** JSON report saved to `data/reports/` (local/backfill)
4. **Optional** audio exported to `exports/`
5. **Database upsert** to `content` + `summaries` (latest variant logic)
6. **Access via Dashboard**; cards appear when a variant has HTML

## 🎵 Audio Delivery Path

1. NAS generates `exports/audio_<video_id>_<timestamp>.mp3` after TTS.
2. NAS uploads the MP3 to the dashboard:
   - Primary: `POST /api/upload-audio` with `Authorization: Bearer $SYNC_SECRET` or `X-INGEST-TOKEN: $INGEST_TOKEN`.
   - Fallback: `POST /ingest/audio` with `X-INGEST-TOKEN`.
   - Server returns JSON including `public_url` (e.g., `/exports/audio/<filename>.mp3`) and `size`. Treat `size==0` as failure and retry.
3. NAS flips Postgres flags/fields on success:
   - `content.has_audio = true`
   - `content.media.audio_url = "/exports/audio/<filename>.mp3"` (root‑relative)
   - `content.media_metadata.mp3_duration_seconds = <int>`
   - `content.audio_version = <unix_ts>` (dashboard appends `?v=`)
4. Dashboard JSON (`/<id>.json`, `/api/reports`) is enriched with `{ kind:'audio', audio_url, duration }` and `has_audio:true`.
5. Health/ops:
   - `GET /health/ingest` shows `token_set` and `pg_dsn_set`.
   - HEAD `https://…/exports/audio/<filename>.mp3?v=<audio_version>` should return 200 with non‑zero `Content-Length`.
   - Disk: ensure sufficient space on `/app/data` (Render) for uploads; server writes with atomic temp→rename to avoid partial files.

## ⚡ Auto‑Process (Idle Run)

Let the bot automatically start a summary after you paste a URL and wait a few seconds. Configure via environment variables:

- `AUTO_PROCESS_DELAY_SECONDS` – enable by setting a positive integer (e.g., `8`). When set, the bot will schedule an auto‑run after the inline keyboard appears. Any tap on the inline buttons cancels the pending auto‑run.
- `AUTO_PROCESS_SUMMARY` – comma‑separated preference list; the first recognized type is chosen. Allowed values: `bullet-points`, `comprehensive`, `key-insights`, `audio`, `audio-fr`, `audio-es`.
- `AUTO_PROCESS_PROVIDER` – comma‑separated provider preferences. Supported values: `ollama`, `cloud`. Aliases: `local`/`hub`/`wireguard` → `ollama`, `api` → `cloud`. The first available is chosen.
  - Ollama availability is probed through the hub (`TTSHUB_API_BASE`). If the hub or its Ollama proxy is unreachable, the bot falls back to `cloud`. Logs show: `AUTO_PROCESS: picked ollama (hub proxy reachable)` or the fallback reason.
- `SUMMARY_TIMEZONE` – time zone name for the timestamp appended to summary headers (default: `America/Denver`).
- `TELEGRAM_SHOW_RESOLVED_PREVIEW` – set to `1`/`true` to post the resolved URL (e.g., expanding `flip.it/…`) so Telegram can show a rich preview before processing.

Notes:
- Provider model selection for auto‑runs uses `QUICK_LOCAL_MODEL` (for `ollama`) or `QUICK_CLOUD_MODEL` (for `cloud`, comma-separated list allowed; first entry is used) when set. Otherwise, it uses the bot’s defaults.
- For Ollama summaries, the bot prefers your TTS hub proxy if `TTSHUB_API_BASE` is set; no `OLLAMA_HOST` is needed for this path.


## 🛠️ Troubleshooting

### Common Issues

- **yt-dlp warnings**: `Requested format is not available` (normal). Metadata falls back to watch-page scraping.
- **Import Errors**: Ensure all essential files are present (nothing left in `archive_nas/`).
- **API Key Issues**: Verify your chosen AI provider key is valid and set in `.env.nas`.
- **DB Write Failures**: Confirm `DATABASE_URL` (or `PG*` vars), role grants (INSERT/UPDATE on `content`, `summaries`), and SSL settings.
- **Docker Issues**: Verify environment file and port availability.

### Log Locations
- **Container Logs**: `docker-compose logs telegram-bot`
- **Bot Activity**: Check Telegram bot responses (multi-part summaries noted)
- **Sync Status**: Monitor dashboard ingest or build a WebSocket/SSE listener for “report created” events
- **Diagnostics**: See `tools/README.md` for targeted scripts (DB tests, ffprobe)

### Back up Portainer environment

- Keep a checked‑in copy of `.env.nas.template` and a private `.env.nas` alongside your stack.
- In Portainer, export your stack (Stacks → your stack → Duplicate/Edit → Copy as text) to capture both the Compose file and env values.
- Optionally, maintain a `runtime.env` inside the container volume (this repo includes one) and source it in your entrypoint; this gives you a quick, human‑readable snapshot after changes.


## 🔐 Codex CLI Authentication (Headless NAS)

Use this flow whenever the Codex CLI requires OAuth on a machine without a local browser (e.g., the NAS). Substitute your own host/IP and SSH port.

1. **Open an SSH tunnel from your Mac**
   ```bash
   ssh -p 1515 -L 1455:localhost:1455 mcdarby2024@24.66.251.193
   ```
   If 1455 is already in use locally, pick another free port (e.g., `1456:localhost:1455`) and adjust the callback in step 3.

2. **Run Codex on the NAS (inside the tunneled session)**
   ```bash
   codex
   ```
   Choose “Sign in with ChatGPT” when prompted. Codex prints a long OAuth URL.

3. **Authenticate in your Mac browser**
   Copy the URL from the NAS terminal, paste it into your Mac’s browser, and sign in to OpenAI. If you used a different local port, replace `localhost:1455` with `localhost:<port>` in the URL before visiting it.

4. **Confirm success**
   The NAS terminal shows `✔ Signed in successfully ...`. Test with `codex hello`.

Repeat the tunnel + login steps any time Codex needs to reauthenticate.

## 🤖 Telegram Bot Actions (Current)

After a summary is generated, the bot presents a three‑row action keyboard designed for clarity on mobile:

- Row 1: `📊 Dashboard` | `📄 Open Summary`
- Row 2: `▶️ Listen` (one‑off) | `🧩 Generate Quiz`
- Row 3: `➕ Add Variant`

Notes:
- “Listen” streams the stored MP3 (generated on the NAS, hosted via `/exports/by_video/<video_id>.mp3`). One‑off TTS remains available through the card actions, but dashboard playback uses the saved variant.
- “Generate Quiz” produces a 10‑item quiz from the Key Points summary (or synthesizes minimal Key Points if missing), optionally categorizes, saves to the Dashboard, and replies with:
  - `▶️ Play in Quizzernator` (deep link, autoplay)
  - `📂 See in Dashboard` (raw JSON)
- Need to delete a summary? Use the dashboard card actions; Telegram deletion will return in a future update.
- The original summary message remains visible; a small status line appears below it while actions are running (⏳/✅).

## 🔌 Dashboard Quiz API dependency

The NAS bot uses these YTV2‑Dashboard endpoints:
- `POST /api/generate-quiz`
- `POST /api/categorize-quiz` (optional; requires `OPENAI_API_KEY` on the Dashboard)
- `POST /api/save-quiz`
- `GET /api/quiz/:filename`

Environment:
- Use `POSTGRES_DASHBOARD_URL` (or `DASHBOARD_BASE_URL`) for Dashboard base URL used by quiz endpoints.

## 🧩 Quizzernator Deep Link

Generated quizzes include a deep link that Quizzernator understands:
- `https://quizzernator.onrender.com/?quiz=api:<filename>&autoplay=1`
- Also accepted: `?quiz=https://<dashboard>/api/quiz/<filename>.json` (auto‑mapped), or `?quiz=<filename>.json`.

## 📝 Prompt Updates (Summary)

Prompts were refined for better structure and TTS quality:
- **Comprehensive**: sections + concise bullets + “Bottom line”
- **Key Points**: 10–16 bullets, ≤ 18 words, concrete facts
- **Key Insights**: 5–7 insights with “— why it matters”; actions
- **Audio**: paragraph‑only narration; “Bottom line”; no headings
- **Chunked** long transcripts: per‑segment bullet summarization
- **Headline**: 12–16 words, no emojis, no colon

## 🔒 Security

- **Environment Variables**: Store API keys securely in `.env.nas`
- **Sync Secret**: Use strong shared secret for Dashboard communication
- **User Access**: Limit Telegram bot to authorized user ID only
- **Network**: Run on private NAS network, sync via HTTPS

### NAS environment workflow

- Portainer is the primary source for env vars; `.env`/`.env.nas` exist only as a synchronized copy for reference.
- After adjusting values in Portainer’s env editor, redeploy from Portainer. Do **not** use `redeploy.sh` (it is disabled) unless you intentionally sync the env files first.
- Keep `.env.nas` in lockstep with Portainer after each change to avoid drift and accidental rollbacks to stale keys/secrets.

## 📊 Performance

- **Concurrent Processing**: Handles one video at a time (prevents resource issues)
- **Duplicate Prevention**: Ledger system avoids reprocessing
- **Storage**: Automatic cleanup of old temporary files
- **Resource Usage**: Optimized for NAS hardware limitations

---

**Part of YTV2 Hybrid Architecture**  
🔗 **Dashboard Component**: See YTV2-Dashboard project for web interface
# Ollama (Local LLM via Hub)

This NAS connects to Ollama running on your Mac through the same hub used for TTS. Set `TTSHUB_API_BASE` on the NAS; no `OLLAMA_URL` needed on the NAS. See docs/ollama.md for endpoints, streaming, and Telegram usage.

### AI↔AI Audio Recap: Gendered Voice Selection

When generating AI↔AI audio recaps with the local TTS hub, personas can include a gender suffix “(M)” or “(F)” (e.g., `Sun Tzu (M)`, `Cleopatra (F)`). The bot strips the suffix for display/prompts and uses it to select voices.

- Enable parsing: `OLLAMA_AI2AI_TTS_GENDER_FROM_PERSONA=1` (default)
- Env selectors:
  - `OLLAMA_AI2AI_TTS_VOICE_MALE` / `OLLAMA_AI2AI_TTS_VOICE_FEMALE`
  - `OLLAMA_AI2AI_TTS_VOICE_A` / `OLLAMA_AI2AI_TTS_VOICE_B` (used when gender not detected)
  - Values accept either a favorite slug (e.g., `favorite--my_voice`) or a catalog `voiceId`.
- Resolution precedence (per speaker):
  1) If gender detected: `…_VOICE_MALE/FEMALE`
  2) First matching favorite by gender
  3) First matching catalog voice by gender
  4) If gender unknown: `…_VOICE_A/B`
  5) Last resort: generic favorites/catalog
- Collision rule: avoids assigning the same `(engine, voiceId)` to both speakers; logs any skip/choice.

Display and prompts strip the “(M)/(F)” suffix for clarity (status text, persona pickers, system prompts, audio intros, captions, and transcripts).

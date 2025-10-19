# YTV2-NAS - YouTube Processing Engine

**The processing component** of the YTV2 hybrid architecture. This runs on your NAS and handles all YouTube video processing, AI summarization, and content generation.

## 🏗️ Architecture Overview

YTV2 uses a **hybrid architecture** with separated concerns:

- **🔧 NAS Component** (This project): YouTube processing + Telegram bot
- **🌐 Dashboard Component**: Web interface + audio playback (deployed to Render)

### How It Works (Postgres-only)

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
- **🗄️ Direct Postgres Writes**: No dashboard upload endpoints; writes happen via UPSERTs
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
   Set `RENDER_DASHBOARD_URL` and `SYNC_SECRET` on the NAS, use the same secret on Render.  
   Each TTS run then pushes MP3s to Postgres (for Listen chips) and to `/app/data/exports/audio/` on Render.

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

### Dashboard Integration (Postgres-only)

The dashboard no longer accepts uploads. NAS writes directly to Postgres using UPSERTs into `content` and `summaries` (latest per `(video_id, variant)`).

Requirements for cards to show:
- At least one summary variant must have non-null `html`
- `language` on `content` is used for language filtering

See `POSTGRES_UPSERT_GUIDE.md` for DDL, indexes, role grants, and UPSERT examples.

### Backfill & Recovery Tools

| Purpose | Command |
| --- | --- |
| Bootstrap schema | `python tools/setup_postgres_schema.py` |
| Connectivity test | `python tools/test_postgres_connect.py` |
| Insert smoke test | `python tools/test_upsert_content.py <video_id>` |
| Resume-safe report ingest | `python tools/backfill_postgres_from_reports.py --resume --audio` |
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
2. `PostgresWriter.upload_content(...)` upserts metadata and HTML-bearing variants.
3. `PostgresWriter.upload_audio(...)` sets `content.has_audio=true` and stores an `<audio>` tag referencing `/exports/audio/<file>.mp3`.
4. NAS uploads the MP3 to Render via `/api/upload-audio` (requires matching `SYNC_SECRET`).
5. Render serves the file via `/exports/by_video/<video_id>.mp3`; dashboard Listen chips stream it instantly.

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
- Row 3: `➕ Add Variant` | `🗑️ Delete…`

Notes:
- “Listen” streams the stored MP3 (generated on the NAS, hosted via `/exports/by_video/<video_id>.mp3`). One‑off TTS remains available through the card actions, but dashboard playback uses the saved variant.
- “Generate Quiz” produces a 10‑item quiz from the Key Points summary (or synthesizes minimal Key Points if missing), optionally categorizes, saves to the Dashboard, and replies with:
  - `▶️ Play in Quizzernator` (deep link, autoplay)
  - `📂 See in Dashboard` (raw JSON)
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

# CLAUDE.md - YTV2-NAS Processing Engine

Important note (2025-10): The dashboard is Postgres-only. Do not call `/ingest/*` HTTP endpoints; write directly to Postgres. Ensure at least one summary variant per video has non-null HTML for card eligibility, and set `content.language` for language filters.

This is the **processing component** of the YTV2 hybrid architecture. It now runs in **Postgres-only mode** (no dual SQLite writes) and handles YouTube video processing, AI summarization (Gemini Flash Lite by default), and content generation on your NAS.

## Project Architecture

This is the **NAS/processing component** of YTV2 that:
- Runs the Telegram bot for user interaction
- Downloads and processes YouTube videos
- Generates AI-powered summaries and analysis
- Creates audio files from video content
- Syncs reports and files to the Dashboard component
- Prevents duplicate processing with ledger system

## Development Commands

### Setup and Installation
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Processing Bot
```bash
# Run the Telegram bot locally (for testing)
python telegram_bot.py

# Or using Docker deployment (recommended for NAS)
docker-compose up -d

# View logs
docker-compose logs -f
```

### Testing
```bash
# Test video processing directly
python youtube_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Test dashboard sync
python nas_sync.py
```

## Architecture Overview

This is the **processing engine** of the YTV2 hybrid architecture:

### YTV2 Hybrid System
- **NAS Component**: This project - Telegram bot + YouTube processing
- **Dashboard Component**: YTV2-Dashboard - Web interface + audio playback

### Processing Pipeline

1. **User Input** → Telegram bot receives YouTube URL
2. **Video Processing** → Download, extract transcript, generate summaries
3. **Content Creation** → JSON reports, audio files, metadata
4. **Duplicate Prevention** → Ledger system checks for existing content
5. **Dashboard Sync** → Upload reports and audio to web interface
6. **User Access** → Web dashboard serves content with audio playback

### Key Components

#### `telegram_bot.py` - Main Processing Bot
- Telegram bot interface for user interactions
- Orchestrates the complete processing pipeline
- Handles user commands, settings, and feedback
- Manages processing queue and status updates

#### `youtube_summarizer.py` - Video Processing Engine
- YouTube video download and transcript extraction
- AI-powered summarization with multiple providers
- Content analysis, categorization, and sentiment detection
- Audio extraction and metadata generation
- Error handling for various video types and restrictions

#### `nas_sync.py` - Dashboard Synchronization
- Writes directly to Postgres (no upload endpoints)
- Coordinates retries and health checks
- Honors feature flags (`POSTGRES_ONLY`, `SQLITE_SYNC_ENABLED`)
- Converts legacy JSON reports when re-syncing (local-only)

#### `export_utils.py` - Content Export
- Multi-format export: JSON, Markdown, HTML, PDF
- File naming and organization
- Template-based report generation
- Export customization and formatting

#### `llm_config.py` - AI Model Management
- Multi-provider LLM support (OpenAI, Anthropic, OpenRouter)
- Model selection and fallback handling  
- API key management and validation
- Configuration loading and environment setup

#### `modules/` - Processing Utilities

**`modules/ledger.py`** - Duplicate Prevention
- JSON-based ledger for tracking processed videos
- Prevents reprocessing of existing content
- Performance optimization for large video collections

**`modules/render_probe.py`** - Dashboard Connectivity  
- Tests Dashboard availability and sync readiness
- Monitors Dashboard health and response times
- Validates sync configuration and credentials

**`modules/telegram_handler.py`** - Bot Interaction Logic
- Telegram message processing and command handling
- User interface components and keyboard layouts
- Session management and state tracking
- Error messaging and user feedback

**`modules/report_generator.py`** - Report Creation
- JSON report structure and validation (only writes to SQLite when explicitly enabled)
- Template processing and data formatting
- Metadata extraction, summary variants, and language tagging

### Data Flow (2025)

1. **Telegram Input** → User sends YouTube URL via bot
2. **Processing** → Video downloaded, transcript fetched, summaries generated (multi-language aware)
3. **Report Generation** → Structured JSON with summary variants & language metadata
4. **Audio Export** → TTS audio generated, chunked if necessary, merged via ffmpeg
5. **Dashboard Sync** → `nas_sync.dual_sync_upload` pushes reports/audio through `PostgresWriter` (SQLite path stays disabled when `POSTGRES_ONLY=true`)
6. **User Access** → Dashboard displays new cards (WebSocket/SSE refresh recommended)

### File Structure

#### Active Processing Files
- `telegram_bot.py` - Main Telegram bot
- `youtube_summarizer.py` - Video processing engine (Gemini Flash Lite default)
- `nas_sync.py` - Dashboard synchronization (Postgres ingest)
- `export_utils.py` - Content export utilities
- `llm_config.py` - AI model configuration & provider fallbacks
- `modules/` - Processing utilities and helpers
  - `modules/services/summary_service.py` consolidates summary orchestration and reprocess logic
  - `modules/services/tts_service.py`, `modules/services/ollama_service.py` keep provider workflows modular
- `tools/` - Diagnostics/test scripts (see `tools/README.md`)
- `data/` - Runtime reports/transcripts (ignored by Git, documented in `data/README.md`)
- `exports/` - Generated audio files (ignored by Git)
- `config/` - Configuration files and templates
- `requirements.txt` - Python dependencies

#### Archived Content
- `archive_nas/old_*` - Previous versions and unused utilities
- `archive/` - Old report backups and historical data

### Environment Configuration

#### Required Variables
```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ADMIN_USER_ID=your_user_id_here

# AI Provider (choose one)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Postgres writer (direct ingest)
DATABASE_URL=postgresql://user:pass@host:5432/ytv2?sslmode=require
AUDIO_PUBLIC_BASE=https://your-render-host  # builds public MP3 links
POSTGRES_ONLY=true  # disables legacy SQLite uploads
SQLITE_SYNC_ENABLED=false  # keep false unless doing forensics

# Optional legacy callbacks (Render API)
SYNC_SECRET=your_shared_secret_here  # only needed if hitting Render /api/upload-audio
```

#### Optional Configuration
```bash
# Processing Settings
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.5-flash-lite
MAX_VIDEO_DURATION=3600
DOWNLOAD_AUDIO_ONLY=false
```

### Deployment

This component is designed for **NAS deployment** using Docker:

- **Docker Compose** for easy container management
- **Volume mounts** for persistent data storage
- **Environment files** for secure configuration
- **Health checks** and automatic restart policies
- **Resource limits** appropriate for NAS hardware

### Integration with Dashboard

The NAS component syncs with the Postgres dashboard via direct database writes:

- UPSERTs into `content` and `summaries` (backing `v_latest_summaries`)
- Ensure one HTML-bearing variant so cards render
- Language written to `content.language` (can also mirror in `analysis_json`)

## PostgreSQL Migration & Legacy Cleanup

### Root Cause Recap (September 2025)
- Auto-running `sync_sqlite_db.py` after each video overwrote the entire Render database, undoing deletions and racing with manual maintenance.
- Maintaining dual SQLite databases (NAS + Render) created brittle variant plumbing and complicated recovery.

### Fixes Applied
- ✅ Migrated to PostgreSQL as the single source of truth (`POSTGRES_ONLY=true`).
- ✅ Added `SQLITE_SYNC_ENABLED` feature flag; defaults to `false` when Postgres-only mode is active.
- ✅ Removed the legacy `sync_sqlite_db.py` calls and all SQLite fallbacks from Telegram/NAS tooling.
- ✅ Hardened ingest (video ID normalisation, audio existence checks, no "most recent file" fallback).
- ✅ Added yt-dlp fallback metadata scraping (channel/duration preserved even when formats are blocked).
- ✅ Propagated summary languages (`summary_language`, `audio_language`, `analysis.language`) so `audio-fr` and `audio-es` variants render correctly on the dashboard.

### Operational Safeguards
- Nightly `backup_database_nas.sh` cron plus Asustor snapshots provide rollbacks.
- Postgres ingest upserts per-video records, so manual deletions on Render persist.
- Always run the Postgres ingest health check before large backfills.

### Next Steps
- Keep `POSTGRES_ONLY=true` in production; only toggle `SQLITE_SYNC_ENABLED=true` for forensic analysis.
- Run dual-sync smoke tests whenever Postgres credentials/config change.

## Important Implementation Notes

- This is **processing-only** - all web serving happens on the Dashboard component
- Uses **Docker deployment** for easy NAS integration
- **Telegram bot** provides the primary user interface
- **AI processing** handles multiple providers with fallback
- **Duplicate prevention** via JSON ledger system for efficiency
- **Audio generation** extracts and exports playable content
- **Dashboard sync** enables web access to all generated content
- **Modular design** separates concerns for maintainability
- **Telemetry**: yt-dlp format warnings are expected; metadata scraping fallback will fill in channel/duration.

**Last Safe State**: PostgreSQL-only sync with Gemini JSON guardrails (September 30, 2025)

## New Capabilities (October 2025)

- Telegram action keyboard (post‑summary) uses a spacious three‑row layout:
  - Row 1: `📊 Dashboard` | `📄 Open Summary`
  - Row 2: `▶️ Listen` (one‑off TTS) | `🧩 Generate Quiz`
  - Row 3: `➕ Add Variant`
- One‑off `Listen` runs chunked TTS with merge, replies with a voice message, and never ingests/saves audio. A small status line appears during work.
- `Generate Quiz` (one‑tap) creates a 10‑item quiz from Key Points (or synthesizes minimal Key Points if missing), optionally categorizes via Dashboard, saves, and replies with deep links:
  - Quizzernator: `https://quizzernator.onrender.com/?quiz=api:<filename>&autoplay=1`
  - Dashboard: `/api/quiz/<filename>`
- Removal of summaries currently happens via the dashboard controls; Telegram deletion will return later if needed.
- Variant discovery for checkmarks now queries the Dashboard (POSTGRES_DASHBOARD_URL) in addition to local JSON/ledger, ensuring all past variants are represented.

### Prompt Refinements

- Comprehensive: concise sections + bullets + “Bottom line”.
- Key Points: 10–16 bullets, ≤ 18 words, concrete facts and names.
- Key Insights: 5–7 insights with “— why it matters”; plus 2–3 actions.
- Audio: paragraph‑only narration, “Bottom line”, no headings/bullets/markdown.
- Chunked transcripts: segment-level bullet prompts; smart combine.
- Headline rule: 12–16 words, no emojis, no colon.

# YTV2 - YouTube Summarization System

Local YouTube processing system running on i9 Mac with Tailscale remote access.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         i9 Mac (Local)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Docker: Bot    │  │ Docker: Dashboard│  │  PostgreSQL     │ │
│  │  Port 6452/6453 │  │  Port 10000     │  │  Port 5432      │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │          │
│           └────────────────────┼────────────────────┘          │
│                                │                               │
│                    ┌───────────┴───────────┐                   │
│                    │     Tailscale VPN     │                   │
│                    └───────────┬───────────┘                   │
└────────────────────────────────┼───────────────────────────────┘
                                 │
┌────────────────────────────────┼───────────────────────────────┐
│                     M4 Mac (Remote)                             │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │  DrawThings     │  │  TTSHUB API     │                      │
│  │  Flux.1 Schnell │  │  Port 7860      │                      │
│  │  (Image Gen)    │  │  (TTS + Images) │                      │
│  └─────────────────┘  └─────────────────┘                      │
│           IP: 100.101.80.13 (Tailscale)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Telegram Bot** | Python/Docker | User interface, URL processing |
| **Dashboard** | Next.js/Docker | Web interface, audio playback |
| **PostgreSQL** | Homebrew | Single source of truth |
| **LLM** | OpenRouter (Gemini 2.5 Flash Lite) | Summarization |
| **TTS** | TTSHUB on M4 Mac | Audio generation |
| **Images** | DrawThings on M4 Mac (Flux.1 Schnell) | Summary thumbnails |

## Image Generation Providers

| Provider | Location | Cost | Quality | Speed |
|----------|----------|------|---------|-------|
| **DrawThings** | M4 Mac (Primary) | Free | Excellent (9/10) | ~5-10s |
| **Z-Image** | Remote | Free | Good | Varies |
| **Flux.2 Klein** | OpenRouter API | $0.014/img | Good | ~5s |
| **Auto1111** | i9 Mac (disabled) | Free | Poor (2/10) | ~6min |

## Quick Start

### Prerequisites
- Docker Desktop
- PostgreSQL (Homebrew)
- Tailscale (for remote access)

### Start Services

```bash
cd /Users/markdarby16/16projects/ytv2/backend

# Start backend
docker-compose up -d

# Check status
docker logs youtube-summarizer-bot --tail 20

# Restart
docker-compose down && docker-compose up -d
```

### Access Points

| Service | Local URL | Tailscale URL |
|---------|-----------|---------------|
| Dashboard | `http://localhost:10000` | `http://marks-macbook-pro-2.tail9e123c.ts.net:10000` |
| Backend API | `http://localhost:6452` | Same via Tailscale |
| Telegram | @Astro74Bot | N/A |

## Configuration

### Environment Variables (`.env.nas`)

```bash
# Database (Local PostgreSQL)
DATABASE_URL=postgresql://ytv2:password@host.docker.internal:5432/ytv2
POSTGRES_ONLY=true
SQLITE_SYNC_ENABLED=false

# Dashboard
DASHBOARD_URL=http://marks-macbook-pro-2.tail9e123c.ts.net:10000

# LLM
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_API_KEY=your_key

# Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_ALLOWED_USERS=your_user_id

# TTS/Images (M4 Mac via Tailscale)
TTSHUB_API_BASE=http://100.101.80.13:7860/api

# Image Generation
SUMMARY_IMAGE_PROVIDERS=drawthings,zimage
SUMMARY_IMAGE_ENABLED=1

# Flux.2 API (optional, paid)
FLUX2_ENABLED=0
```

## CLI Commands

```bash
# Status check
./ytv2 status
./ytv2.          # Menu

# Logs
docker logs youtube-summarizer-bot --tail 50 -f

# Database
python tools/test_postgres_connect.py
python tools/debug_audio_variant.py <video_id>
```

## Bot Commands (Telegram)

| Command | Description |
|---------|-------------|
| `/status` | System health snapshot |
| `/diag` | Diagnostics |
| `/logs [N]` | Recent log lines |
| `/restart` | Restart container |
| `/draw <prompt>` | Image generation helper |

## File Structure

```
backend/
├── .env.nas              # Environment config (source of truth)
├── docker-compose.yml    # Docker config
├── telegram_bot.py       # Main bot
├── youtube_summarizer.py # Processing engine
├── nas_sync.py          # Database sync
├── modules/
│   ├── services/        # TTS, images, LLM services
│   ├── ledger.py        # Duplicate prevention
│   └── telegram_handler.py
├── tools/               # Diagnostic scripts
├── data/                # Runtime reports
└── exports/             # Generated audio/images
```

## Image Generation Services

### DrawThings (M4 Mac) - Primary
- Model: Flux.1 Schnell
- Steps: 6
- Resolution: 384x384
- Free, high quality

### Flux.2 Klein 4B (OpenRouter) - Optional Fallback
- Cost: $0.014/image (~$0.07/summary)
- Enable: `FLUX2_ENABLED=1` and add `flux2` to `SUMMARY_IMAGE_PROVIDERS`
- Speed: ~5 seconds

### Auto1111 (i9 Mac) - Disabled
- Quality too poor for use
- SDXL on CPU produces unusable images

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
docker-compose down
docker-compose up -d
docker logs youtube-summarizer-bot
```

**Database connection failed:**
```bash
# Check PostgreSQL is running
brew services list
brew services start postgresql@14

# Test connection
python tools/test_postgres_connect.py
```

**Image generation failing:**
- Check M4 Mac is reachable: `curl http://100.101.80.13:7860/api/health`
- Verify Tailscale connection

### Log Locations
- Container: `docker logs youtube-summarizer-bot`
- Bot file: `backend/bot.log`

## Remote Access (Tailscale)

The system is accessible remotely via Tailscale:
- Dashboard: `http://marks-macbook-pro-2.tail9e123c.ts.net:10000`
- Works from iPhone, iPad, other Macs

## Migration History

| Date | Change |
|------|--------|
| Feb 2026 | Migrated from NAS/Render to i9 Mac local |
| Feb 2026 | Added Flux.2 Klein 4B support (disabled) |
| Feb 2026 | Disabled Auto1111 (poor CPU quality) |
| Sep 2025 | Migrated to PostgreSQL-only |

---

**Previous Architecture (Legacy):**
- NAS deployment via Portainer
- Render cloud hosting
- SQLite databases

See `archive/` for historical documentation.

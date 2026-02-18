# Local Integration (i9 Mac)

This document describes the current local deployment architecture.

> **Note:** This was previously NAS_INTEGRATION.md for Render deployment. The system now runs locally on i9 Mac.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           i9 Mac                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Docker Containers                     │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │ Backend Bot     │  │ Dashboard                   │   │   │
│  │  │ Port 6452/6453  │  │ Port 10000                  │   │   │
│  │  │ - Telegram Bot  │  │ - Web Interface             │   │   │
│  │  │ - Processing    │  │ - Audio Playback            │   │   │
│  │  └────────┬────────┘  └──────────────┬──────────────┘   │   │
│  │           │                          │                   │   │
│  │           └────────────┬─────────────┘                   │   │
│  │                        │                                 │   │
│  │              host.docker.internal                        │   │
│  └────────────────────────┼─────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │              PostgreSQL (Homebrew)                       │   │
│  │              Port 5432                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│                    ┌──────────────────┐                         │
│                    │    Tailscale     │                         │
│                    └────────┬─────────┘                         │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Tailscale VPN   │
                    └─────────┬─────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                     M4 Mac   │                                   │
│                    (Remote)  │                                   │
│  ┌──────────────────────────┴──────────────────────────────┐   │
│  │                    TTSHUB API                            │   │
│  │                    Port 7860                             │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │ DrawThings      │  │ TTS Services                │   │   │
│  │  │ Flux.1 Schnell  │  │ - Audio Generation          │   │   │
│  │  │ Image Gen       │  │ - Voice Synthesis           │   │   │
│  │  └─────────────────┘  └─────────────────────────────┘   │   │
│  │                                                         │   │
│  │  IP: 100.101.80.13 (Tailscale)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### i9 Mac (Local Host)

| Service | Port | Description |
|---------|------|-------------|
| Backend Bot | 6452 | Telegram bot, `/api/reprocess`, sync endpoints |
| Backend API | 6453 | YTV2 API server (Clawdbot) - `/api/process`, `/api/transcript` |
| Dashboard | 10000 | Web interface |
| PostgreSQL | 5432 | Database |

> **IMPORTANT:** The backend container runs **TWO HTTP servers**:
> - **Port 6452**: `telegram_bot.py` - has `/api/reprocess` (dashboard connects here!)
> - **Port 6453**: FastAPI (`ytv2_api/`) - video processing endpoints
>
> Dashboard's `BACKEND_API_URL` should point to **port 6452** for regenerate functionality.

### M4 Mac (Remote via Tailscale)

| Service | Port | Description |
|---------|------|-------------|
| TTSHUB API | 7860 | TTS and image generation |
| DrawThings | - | Flux.1 Schnell image generation |

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://ytv2:password@host.docker.internal:5432/ytv2
POSTGRES_ONLY=true

# Dashboard
DASHBOARD_URL=http://marks-macbook-pro-2.tail9e123c.ts.net:10000

# Auth for regenerate endpoint (must match dashboard's DEBUG_TOKEN)
REPROCESS_AUTH_TOKEN=your_shared_secret

# Sync secret (must match dashboard's SYNC_SECRET)
SYNC_SECRET=your_sync_secret

# M4 Mac Services (via Tailscale)
TTSHUB_API_BASE=http://100.101.80.13:7860/api

# Image Generation
SUMMARY_IMAGE_PROVIDERS=drawthings,zimage
SUMMARY_IMAGE_ENABLED=1

# Flux.2 API (optional, paid fallback)
FLUX2_ENABLED=0
```

### Docker Networking

The backend container connects to:
- **PostgreSQL**: `host.docker.internal:5432`
- **M4 Mac**: `100.101.80.13:7860` (Tailscale IP)

## Database

### PostgreSQL Setup (Homebrew)

```bash
# Install
brew install postgresql@14
brew services start postgresql@14

# Create database
createdb ytv2
psql ytv2 -c "CREATE USER ytv2 WITH PASSWORD 'password';"
psql ytv2 -c "GRANT ALL PRIVILEGES ON DATABASE ytv2 TO ytv2;"

# Schema setup
python tools/setup_postgres_schema.py
```

### Connection Test

```bash
python tools/test_postgres_connect.py
```

## Image Generation

### DrawThings (M4 Mac) - Primary

- Model: Flux.1 Schnell
- Steps: 6
- Resolution: 384x384
- Quality: Excellent (9/10)
- Cost: Free

### Flux.2 Klein 4B (OpenRouter) - Optional

- Cost: $0.014/image
- Enable: `FLUX2_ENABLED=1`
- Add `flux2` to `SUMMARY_IMAGE_PROVIDERS`

### Auto1111 (i9 Mac) - Disabled

SDXL on CPU produces poor quality images (2/10). Not recommended.

## Remote Access

### Tailscale Setup

1. Install Tailscale on both Macs
2. Ensure both are connected to same tailnet
3. Verify connectivity: `ping 100.101.80.13`

### Access URLs

| From | To | URL |
|------|-----|-----|
| M4 Mac | Dashboard | `http://marks-macbook-pro-2.tail9e123c.ts.net:10000` |
| i9 Mac | M4 Mac TTSHUB | `http://100.101.80.13:7860/api` |
| iPhone/iPad | Dashboard | Same Tailscale URL |

## Troubleshooting

### M4 Mac Unreachable

```bash
# Check Tailscale
tailscale status

# Test connectivity
curl http://100.101.80.13:7860/api/health
```

### Database Connection Failed

```bash
# Check PostgreSQL
brew services list
brew services restart postgresql@14

# Test from container
docker exec youtube-summarizer-bot python -c "
import os
import psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
print('Connected!')
"
```

### Image Generation Failing

1. Check M4 Mac is online and Tailscale connected
2. Verify TTSHUB is running on M4 Mac
3. Check health: `curl http://100.101.80.13:7860/api/health`

## Migration History

| Date | Change |
|------|--------|
| Feb 2026 | Migrated from NAS to i9 Mac local |
| Feb 2026 | Added Flux.2 Klein support |
| Feb 2026 | Disabled Auto1111 (poor quality) |
| Sep 2025 | Migrated to PostgreSQL-only |

# CLAUDE.md - YTV2 Backend

YouTube summarization system running locally on i9 Mac with Tailscale remote access.

> **See also:** [../ARCHITECTURE.md](../ARCHITECTURE.md) for full system architecture diagram.

## Current Architecture

```
i9 Mac (Local Host)
├── Docker: youtube-summarizer-bot (Port 6452/6453)
├── Docker: Dashboard (Port 10000)
├── PostgreSQL via Homebrew (Port 5432)
└── Tailscale VPN for remote access

M4 Mac (Remote via Tailscale: 100.101.80.13)
├── DrawThings (Flux.1 Schnell image generation)
└── TTSHUB API (Port 7860)
```

## ⚠️ IMPORTANT: Two HTTP Servers

This container runs **TWO separate HTTP servers** on different ports:

| Port | Server | Purpose |
|------|--------|---------|
| **6452** | `telegram_bot.py` | Telegram bot, `/api/reprocess`, sync endpoints |
| **6453** | FastAPI (`ytv2_api/`) | Video processing `/api/process`, `/api/transcript` |

**Dashboard connects to port 6452** for the regenerate functionality!

## Quick Commands

```bash
# Start/restart
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose down && docker-compose up -d

# Logs
docker logs youtube-summarizer-bot --tail 50 -f

# Status CLI
./ytv2 status
```

## Environment (`.env.nas`)

Key variables:
```bash
DATABASE_URL=postgresql://ytv2:pass@host.docker.internal:5432/ytv2
POSTGRES_ONLY=true
DASHBOARD_URL=http://marks-macbook-pro-2.tail9e123c.ts.net:10000
TTSHUB_API_BASE=http://100.101.80.13:7860/api
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.5-flash-lite
SUMMARY_IMAGE_PROVIDERS=drawthings,zimage
FLUX2_ENABLED=0
```

## Image Generation

| Provider | Status | Location | Notes |
|----------|--------|----------|-------|
| DrawThings | Primary | M4 Mac | Flux.1 Schnell, free, excellent quality |
| Z-Image | Backup | Remote | Free |
| Flux.2 Klein | Disabled | OpenRouter API | $0.014/image, enable with FLUX2_ENABLED=1 |
| Auto1111 | Disabled | i9 Mac | SDXL on CPU produces poor quality |

## File Structure

```
backend/
├── telegram_bot.py       # Main bot entry
├── youtube_summarizer.py # Video processing
├── nas_sync.py          # DB sync
├── modules/services/    # TTS, image, LLM services
│   ├── draw_service.py
│   ├── flux2_service.py
│   ├── auto1111_service.py
│   └── summary_image_service.py
├── tools/               # Diagnostics
└── .env.nas            # Config (source of truth)
```

## Services

### flux2_service.py
- Flux.2 Klein 4B via OpenRouter
- Cost: $0.014/image
- Gated by `FLUX2_ENABLED=1`

### auto1111_service.py
- Automatic1111 SDXL on i9 Mac
- Disabled due to poor CPU quality
- 6+ min generation, 2/10 quality

### draw_service.py
- DrawThings on M4 Mac via TTSHUB
- Primary image provider
- Flux.1 Schnell, ~5-10s generation

## Database

PostgreSQL-only mode (no SQLite):
- `POSTGRES_ONLY=true`
- `SQLITE_SYNC_ENABLED=false`
- Direct connection via `host.docker.internal:5432`

## Remote Access

- Tailscale URL: `http://marks-macbook-pro-2.tail9e123c.ts.net:10000`
- M4 Mac IP: `100.101.80.13`

## Bot Commands

- `/status` - Health snapshot
- `/diag` - Diagnostics
- `/logs [N]` - Recent logs
- `/restart` - Restart container

## Important Notes

1. **PostgreSQL-only** - No SQLite writes
2. **Images from M4 Mac** - DrawThings via TTSHUB
3. **Tailscale required** - For M4 Mac communication
4. **Flux.2 gated** - Enable only if willing to pay

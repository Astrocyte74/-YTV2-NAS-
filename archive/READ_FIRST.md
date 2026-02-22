# READ FIRST

YTV2 is a YouTube summarization system running locally on i9 Mac with remote access via Tailscale.

## Current Architecture

| Component | Location | Access |
|-----------|----------|--------|
| **Backend** | i9 Mac Docker | `localhost:6452` |
| **Dashboard** | i9 Mac Docker | `localhost:10000` / Tailscale |
| **PostgreSQL** | i9 Mac (Homebrew) | `host.docker.internal:5432` |
| **Telegram Bot** | i9 Mac Docker | @Astro74Bot |
| **TTS/Images** | M4 Mac (DrawThings) | `100.101.80.13:7860` (Tailscale) |

## Quick Start

```bash
# Start/restart services
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose down && docker-compose up -d

# Check logs
docker logs youtube-summarizer-bot --tail 50 -f

# Status via CLI
./ytv2 status
```

## Key Files

| File | Purpose |
|------|---------|
| `.env.nas` | All environment variables (source of truth) |
| `docker-compose.yml` | Docker configuration |
| `telegram_bot.py` | Main bot entry point |
| `modules/services/` | Service modules (TTS, images, LLM) |

## Remote Access

- **Tailscale URL**: `http://marks-macbook-pro-2.tail9e123c.ts.net:10000`
- **Dashboard**: Port 10000
- **Backend API**: Port 6452

## Documentation

- `README.md` - Full documentation
- `docs/` - Additional guides
- `tools/README.md` - Diagnostic scripts

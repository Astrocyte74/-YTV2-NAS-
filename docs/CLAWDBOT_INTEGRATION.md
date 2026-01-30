# Clawdbot Multi-Channel Gateway Integration

This guide explains how to integrate YTV2 with Clawdbot, enabling URL processing from multiple messaging platforms including WhatsApp, Discord, Slack, iMessage, and more.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Multi-Channel Input                           │
│  Telegram │ WhatsApp │ Discord │ Slack │ iMessage │ WebChat      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Clawdbot     │  ← Multi-channel gateway
                    │   Gateway      │     (Node.js on NAS)
                    └────────┬───────┘
                             │
                             │ HTTP POST (localhost)
                             ▼
                    ┌────────────────┐
                    │     YTV2       │  ← Processing engine
                    │  API Server    │     (Python on NAS)
                    └────────┬───────┘
                             │
                    ┌────────▼───────┐
                    │  YouTube/      │
                    │  Reddit/Web    │  ← Existing processing
                    │  Processing    │     (GLM/OpenAI/etc)
                    └────────────────┘
                             │
                             ▼
                    Reply via original channel
```

## Prerequisites

1. **YTV2** is already installed and running on your NAS
2. **Docker** and **Docker Compose** are available
3. **Network access** between containers

## Step 1: Enable YTV2 API Server

### Option A: Via Docker Compose (Recommended)

Edit `docker-compose.yml` and set `YTV2_API_ENABLED=true`:

```yaml
services:
  youtube-summarizer-bot:
    environment:
      # ... existing config ...
      - YTV2_API_ENABLED=true
      - YTV2_API_PORT=6453
      - YTV2_API_HOST=0.0.0.0  # Bind to all interfaces for Clawdbot access
```

### Option B: Via .env.nas File

Add these lines to your `.env.nas`:

```bash
# YTV2 API Server
YTV2_API_ENABLED=true
YTV2_API_PORT=6453
YTV2_API_HOST=0.0.0.0
YTV2_API_SECRET=your-secure-random-secret-here  # Optional but recommended
```

### Optional: Set API Secret (Recommended)

Generate a secure random secret:

```bash
# Generate random secret
openssl rand -hex 32

# Add to .env.nas
YTV2_API_SECRET=<generated-secret>
```

### Restart YTV2

```bash
cd /volume1/Docker/YTV2
docker compose down
docker compose up -d
```

### Verify API is Running

```bash
# From within NAS
curl http://localhost:6453/health

# Expected response:
# {"status":"healthy","service":"ytv2-api","version":"1.0.0"}
```

## Step 2: Install Clawdbot

### Create Clawdbot Directory

```bash
mkdir -p /volume1/Docker/clawdbot
cd /volume1/Docker/clawdbot
```

### Create Docker Compose File

Create `docker-compose.yml`:

```yaml
services:
  clawdbot-gateway:
    image: clawdbot/clawdbot:latest
    container_name: clawdbot-gateway
    restart: unless-stopped
    ports:
      - "18789:18789"  # Gateway WebSocket
    volumes:
      - ~/.clawdbot:/root/.clawdbot
      - ~/clawd:/root/clawd
    environment:
      - NODE_ENV=production
      - GATEWAY_BIND=0.0.0.0
      - GATEWAY_PORT=18789
    networks:
      - ytv2-network

networks:
  ytv2-network:
    external: true
```

### Connect to YTV2 Network

```bash
# Connect Clawdbot to existing YTV2 network
docker network connect ytv2-network clawdbot-gateway
```

### Start Clawdbot

```bash
docker compose up -d
```

### Run Onboarding Wizard

```bash
docker compose exec clawdbot-gateway clawdbot onboard
```

Follow the prompts to configure your gateway.

## Step 3: Configure Channels

### Telegram (Existing Bot)

Clawdbot can use your existing Telegram bot token:

```bash
docker compose exec clawdbot-gateway clawdbot channels add \
  --channel telegram \
  --token "$TELEGRAM_BOT_TOKEN"
```

### WhatsApp (New - QR Code Required)

```bash
docker compose exec clawdbot-gateway clawdbot channels login
# Select WhatsApp and scan QR code with your phone
```

### Discord (Optional - New Bot Token)

1. Create a Discord bot at https://discord.com/developers/applications
2. Get the bot token
3. Add to Clawdbot:

```bash
docker compose exec clawdbot-gateway clawdbot channels add \
  --channel discord \
  --token "$DISCORD_BOT_TOKEN"
```

### Slack (Optional - OAuth Required)

Follow Clawdbot's Slack setup guide via the onboarding wizard.

### iMessage (Mac Only)

Requires running Clawdbot on a Mac with Messages access.

## Step 4: Install YTV2 Integration Skill

### Copy Skill Files

The YTV2 skill files are located in:
- `/root/.clawdbot/skills/ytv2-integration/SKILL.md`
- `/root/.clawdbot/skills/ytv2-integration/ytv2.sh`

If installing Clawdbot in a different location, copy these files to:

```bash
~/.clawdbot/skills/ytv2-integration/SKILL.md
~/.clawdbot/skills/ytv2-integration/ytv2.sh
chmod +x ~/.clawdbot/skills/ytv2-integration/ytv2.sh
```

### Configure Skill Environment

Edit `~/.clawdbot/clawdbot.json`:

```json
{
  "skills": {
    "entries": {
      "ytv2-integration": {
        "enabled": true,
        "env": {
          "YTV2_API_URL": "http://youtube-summarizer-bot:6453",
          "YTV2_API_SECRET": "your-secure-random-secret-here"
        }
      }
    }
  }
}
```

### Restart Clawdbot

```bash
docker compose restart clawdbot-gateway
```

## Step 5: Testing

### Test YouTube URL

Send a YouTube URL from any configured channel:

```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Expected response:
```
📄 <Video Title>

<Summary text>

---
Source: YouTube • Duration: X:XX • Channel: <Channel Name>
Dashboard: https://your-dashboard.com/watch/abc123
```

### Test Reddit URL

```
https://reddit.com/r/technology/comments/example
```

### Test Web Article

```
https://example.com/article
```

### Test API Directly

```bash
curl -X POST "http://localhost:6453/api/process" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $YTV2_API_SECRET" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "summary_type": "comprehensive",
    "user_id": "test",
    "channel": "telegram"
  }'
```

## Configuration Reference

### YTV2 Environment Variables (.env.nas)

| Variable | Default | Description |
|----------|---------|-------------|
| `YTV2_API_ENABLED` | `false` | Enable API server |
| `YTV2_API_HOST` | `127.0.0.1` | API bind address |
| `YTV2_API_PORT` | `6453` | API port |
| `YTV2_API_SECRET` | (none) | Optional auth token |
| `DASHBOARD_URL` | (none) | Dashboard base URL |

### Clawdbot Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `YTV2_API_URL` | Yes | Full API URL |
| `YTV2_API_SECRET` | Optional | Auth token if configured |

## Troubleshooting

### API Not Reachable from Clawdbot

**Problem**: Clawdbot cannot connect to YTV2 API

**Solutions**:
1. Check YTV2 is running: `docker compose logs youtube-summarizer-bot`
2. Check API is enabled: `docker compose exec youtube-summarizer-bot env | grep YTV2_API`
3. Check network connectivity:
   ```bash
   docker network inspect ytv2-network
   docker compose exec clawdbot-gateway ping youtube-summarizer-bot
   ```
4. If using `127.0.0.1`, change to `0.0.0.0` for YTV2_API_HOST

### Authentication Errors

**Problem**: 401 Unauthorized responses

**Solution**:
- Verify `YTV2_API_SECRET` matches in both YTV2 and Clawdbot
- Check Authorization header format: `Bearer <token>`

### Reddit Processing Fails

**Problem**: Reddit URLs return credential errors

**Solution**:
Configure Reddit credentials in `.env.nas`:
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_REFRESH_TOKEN=your_refresh_token
REDDIT_USER_AGENT=your_user_agent
```

### Web Article Extraction Fails

**Problem**: Web URLs return extraction errors

**Possible causes**:
- Site blocks automated access
- JavaScript-heavy content (requires browser)
- Paywall or login required

**Solution**: Try a different URL or copy content directly.

### Large Content Timeout

**Problem**: Long videos timeout before processing completes

**Solutions**:
1. Increase timeout in Clawdbot skill configuration
2. Consider splitting long content
3. Use `summary_type: bullet-points` for faster processing

## Security Considerations

1. **Network Isolation**: Both services run on same Docker network
2. **Authentication**: Use `YTV2_API_SECRET` for inter-service communication
3. **No External Exposure**: API binds to localhost/internal network only
4. **Rate Limiting**: Consider implementing rate limits for abuse prevention

## Rollback Plan

If issues arise:

1. **Stop Clawdbot**:
   ```bash
   cd /volume1/Docker/clawdbot
   docker compose down
   ```

2. **Disable YTV2 API**:
   Set `YTV2_API_ENABLED=false` in `.env.nas` or `docker-compose.yml`

3. **Restart YTV2**:
   ```bash
   cd /volume1/Docker/YTV2
   docker compose restart
   ```

The original Telegram bot continues working independently throughout.

## Additional Resources

- **Clawdbot Documentation**: https://clawdbot.dev/docs
- **YTV2 Documentation**: See `CLAUDE.md` in this repository
- **API Examples**: See `docs/API_EXAMPLES.md`

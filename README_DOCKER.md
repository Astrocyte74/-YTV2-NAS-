# YTV2 Backend Docker Deployment

## Critical: Start from Correct Directory!

The backend container MUST be started from the **ytv2/backend** directory, NOT from any other directory.

### Why This Matters

The container uses `./:/app` volume mount, which mounts the current directory to `/app` in the container. If started from the wrong directory:
- JSON reports won't be accessible to the sync code
- PostgreSQL writes will fail silently
- Telegram summaries won't appear in dashboard

### Correct Startup Procedure

```bash
# 1. Navigate to backend directory
cd /Users/markdarby16/16projects/ytv2/backend

# 2. Verify you're in the right place
pwd  # Should show: /Users/markdarby16/16projects/ytv2/backend
ls docker-compose.yml  # Should exist

# 3. Start container
docker-compose up -d

# 4. Verify mounts
docker inspect youtube-summarizer-bot --format='{{json .Mounts}}' | grep '"Source"'
# Should show: /Users/markdarby16/16projects/ytv2/backend
```

### Safety Checks Added

1. **Project Name**: docker-compose.yml now has `name: ytv2-backend` to prevent conflicts
2. **Entrypoint Validation**: Container will fail to start if /app/data is not writable
3. **Health Check**: Container health check verifies data directory accessibility

### Troubleshooting

If summaries aren't appearing in dashboard:

```bash
# Check where container was started from
docker inspect youtube-summarizer-bot --format='{{.Config.Labels}}' | grep working_dir

# Check mounts
docker inspect youtube-summarizer-bot --format='{{json .Mounts}}' | python3 -m json.tool

# Restart from correct directory if wrong
docker stop youtube-summarizer-bot && docker rm youtube-summarizer-bot
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose up -d
```

### Preventing Future Issues

- NEVER start with `docker run` directly - always use `docker-compose` from ytv2/backend
- The entrypoint will fail fast if mounts are incorrect
- Check logs for "[entrypoint] ❌ ERROR" messages

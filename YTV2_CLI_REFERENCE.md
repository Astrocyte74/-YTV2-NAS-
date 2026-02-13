# YTV2 CLI - Quick Reference

## Installation
The CLI is installed at `~/bin/ytv2` and added to your `~/.zshrc`.

**After installing, restart your shell or run:**
```bash
source ~/.zshrc
```

## Two Ways to Use YTV2

### 1. Direct Commands (fast)
```bash
ytv2 status    # Show status
ytv2 stats     # Show statistics
ytv2 restart   # Restart servers
ytv2 logs      # Show logs
ytv2 verify    # Verify setup
```

### 2. Interactive Menu (visual)
```bash
ytv2.    # Opens interactive menu (with period)
```

The interactive menu shows:
- Live status panels (health indicators)
- Statistics (reports, queues, storage)
- Access URLs
- Action menu with numbered choices

## Commands

### `ytv2 status` (default)
Shows server status and access URLs.
```bash
ytv2
# or
ytv2 status
```

**Output:**
- Backend status (healthy/unhealthy)
- Dashboard status
- PostgreSQL status
- Mount verification (catches wrong-directory issues!)
- Access URLs (localhost & Tailscale)
- Recent activity (last 5 PROCESSING/Syncing/Error messages)

### `ytv2 stats`
Shows dashboard statistics.
```bash
ytv2 stats
```

**Output:**
- Total reports in database
- Reports added in last 24h
- Breakdown by type (YouTube vs Web articles)
- Latest entry title
- Image queue status (pending jobs)
- TTS queue status (pending jobs)
- Storage usage (backend data, exports, dashboard exports)

### `ytv2 restart`
Safely restarts all servers (URLs don't change!).
```bash
ytv2 restart
```

**What it does:**
1. Stops backend and dashboard containers
2. Starts backend from correct directory (`ytv2/backend`)
3. Starts dashboard from correct directory (`dashboard16`)
4. Verifies mounts are correct
5. Checks health status
6. Displays access URLs

### `ytv2 logs [service] [n]`
Shows recent logs.
```bash
ytv2 logs                    # Backend logs, 50 lines
ytv2 logs ytv2-dashboard     # Dashboard logs
ytv2 logs youtube-summarizer-bot 100  # Backend, 100 lines
```

### `ytv2 verify`
Verifies setup and detects issues.
```bash
ytv2 verify
```

**Checks:**
- Backend directory exists
- Dashboard directory exists
- PostgreSQL is running
- Backend container has correct mount (ytv2/backend)
- Dashboard container is running
- Dashboard is responding

### `ytv2 help`
Shows help message.
```bash
ytv2 help
```

## Examples

```bash
# Check if everything is running
ytv2

# See how many reports you have
ytv2 stats

# Restart servers after config change
ytv2 restart

# Check for issues
ytv2 verify

# See what the backend is doing
ytv2 logs
```

## Troubleshooting

**"command not found: ytv2"**
```bash
# Add to current session
export PATH="$HOME/bin:$PATH"

# Or restart shell
source ~/.zshrc
```

**"Backend container mount is WRONG"**
```bash
# Run restart to fix
ytv2 restart
```

**"Dashboard is not responding"**
```bash
# Check logs
ytv2 logs ytv2-dashboard

# Restart
ytv2 restart
```

# YTV2-NAS - YouTube Processing Engine

**The processing component** of the YTV2 hybrid architecture. This runs on your NAS and handles all YouTube video processing, AI summarization, and content generation.

## 🏗️ Architecture Overview

YTV2 uses a **hybrid architecture** with separated concerns:

- **🔧 NAS Component** (This project): YouTube processing + Telegram bot
- **🌐 Dashboard Component**: Web interface + audio playback (deployed to Render)

### How It Works

1. **📱 Telegram Bot** receives YouTube URLs from users
2. **🤖 AI Processing** downloads, transcribes, and summarizes videos  
3. **📊 JSON Reports** generated with metadata and summaries
4. **🎵 Audio Export** creates playable audio files
5. **🔄 Auto-Sync** uploads reports and audio to Dashboard
6. **🌐 Web Access** users view summaries and play audio via Dashboard URL

## ✨ Features

- **🤖 Telegram Bot Interface**: Send YouTube URLs for instant AI processing
- **🎯 AI-Powered Summarization**: Multiple summary types with sentiment analysis
- **🔄 Duplicate Prevention**: JSON ledger system prevents reprocessing videos
- **🎵 Audio Generation**: Extract and export audio with metadata
- **📊 Structured Reports**: JSON format with rich metadata
- **🌐 Dashboard Sync**: Automatic upload to web dashboard
- **⚙️ Multi-AI Support**: OpenAI, Anthropic Claude, OpenRouter
- **🔧 Docker Ready**: Easy NAS deployment via Portainer

## 🚀 Quick Setup

### Prerequisites

- **Docker** environment (Portainer recommended)
- **API Keys**: OpenAI, Anthropic, or OpenRouter
- **Telegram Bot**: Token from @BotFather
- **Dashboard URL**: Your YTV2-Dashboard Render deployment

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
   
   # Dashboard Sync
   RENDER_DASHBOARD_URL=your_dashboard_url_here
   SYNC_SECRET=your_shared_secret_here
   ```

4. **Deploy with Docker**:
   ```bash
   docker-compose up -d
   ```

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

### Dashboard Integration

Connect to your YTV2-Dashboard deployment:

```bash
# Your dashboard URL (Render deployment)
RENDER_DASHBOARD_URL=https://your-dashboard.onrender.com

# Shared secret for secure sync
SYNC_SECRET=your_secure_secret_here
```

## 📁 Project Structure

### Essential Files
```
YTV2-NAS/
├── telegram_bot.py          # Main Telegram bot
├── youtube_summarizer.py    # Video processing engine  
├── nas_sync.py             # Dashboard synchronization
├── export_utils.py         # Summary export utilities
├── llm_config.py          # AI model configuration
├── modules/               # Processing utilities
│   ├── ledger.py         # Duplicate prevention
│   ├── render_probe.py   # Dashboard connectivity  
│   └── telegram_handler.py # Bot interaction logic
├── data/                  # Generated JSON reports
├── exports/              # Audio files and exports
└── config/              # Configuration files
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

1. **Send YouTube URL** to your Telegram bot
2. **Bot processes** video (download → transcribe → summarize)
3. **Reports generated** in `data/reports/` as JSON
4. **Audio exported** to `exports/` directory  
5. **Auto-sync** uploads to Dashboard
6. **Access via Dashboard** URL for web viewing

## 🛠️ Troubleshooting

### Common Issues

- **Import Errors**: Ensure all essential files are present (not archived)
- **API Key Issues**: Verify your chosen AI provider key is valid
- **Sync Failures**: Check Dashboard URL and shared secret
- **Docker Issues**: Verify environment file and port availability

### Log Locations
- **Container Logs**: `docker-compose logs telegram-bot`
- **Bot Activity**: Check Telegram bot responses
- **Sync Status**: Monitor Dashboard for new reports

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
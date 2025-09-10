# CLAUDE.md - YTV2-NAS Processing Engine

This is the **processing component** of the YTV2 hybrid architecture - it handles YouTube video processing, AI summarization, and content generation on your NAS.

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
- Uploads JSON reports to Dashboard component
- Syncs audio files and exports
- Handles authentication and error recovery
- Maintains sync status and retry logic

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
- JSON report structure and validation
- Template processing and data formatting
- Metadata extraction and organization

### Data Flow

1. **Telegram Input** → User sends YouTube URL via bot
2. **Processing** → Video downloaded, transcribed, summarized by AI
3. **Report Generation** → Structured JSON with metadata and summaries
4. **Audio Export** → Audio files extracted with proper metadata
5. **Dashboard Sync** → Files uploaded to web component
6. **User Access** → Web dashboard displays reports with audio playback

### File Structure

#### Active Processing Files
- `telegram_bot.py` - Main Telegram bot
- `youtube_summarizer.py` - Video processing engine
- `nas_sync.py` - Dashboard synchronization
- `export_utils.py` - Content export utilities
- `llm_config.py` - AI model configuration
- `modules/` - Processing utilities and helpers
- `data/` - Generated JSON reports
- `exports/` - Audio files and other exports
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

# Dashboard Integration
RENDER_DASHBOARD_URL=your_dashboard_url_here
SYNC_SECRET=your_shared_secret_here
```

#### Optional Configuration
```bash
# Processing Settings
MAX_VIDEO_DURATION=3600
DOWNLOAD_AUDIO_ONLY=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
```

### Deployment

This component is designed for **NAS deployment** using Docker:

- **Docker Compose** for easy container management
- **Volume mounts** for persistent data storage
- **Environment files** for secure configuration
- **Health checks** and automatic restart policies
- **Resource limits** appropriate for NAS hardware

### Integration with Dashboard

The NAS component syncs with the Dashboard component via:

- **HTTP API** for uploading reports and files
- **Shared secrets** for authentication
- **JSON format** for structured data exchange
- **Error handling** with retry logic for network issues

## Important Implementation Notes

- This is **processing-only** - all web serving happens on the Dashboard component
- Uses **Docker deployment** for easy NAS integration
- **Telegram bot** provides the primary user interface
- **AI processing** handles multiple providers with fallback
- **Duplicate prevention** via JSON ledger system for efficiency
- **Audio generation** extracts and exports playable content
- **Dashboard sync** enables web access to all generated content
- **Modular design** separates concerns for maintainability
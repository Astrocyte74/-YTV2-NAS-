#!/bin/bash
# Main startup script for YouTube Summarizer Bot container
# Installs system and Python dependencies, then starts the bot

set -e  # Exit on any error

echo "🚀 Starting YouTube Summarizer Bot container setup..."

# Install system dependencies (ffmpeg, etc.)
if [ -f "./install_deps.sh" ]; then
    echo "📦 Installing system dependencies..."
    ./install_deps.sh
else
    echo "⚠️ install_deps.sh not found, skipping system dependencies"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Start the bot
echo "🤖 Starting YouTube Summarizer Bot..."
python telegram_bot.py
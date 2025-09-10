#!/bin/bash
# System dependencies installation script for YouTube Summarizer Bot
# This script installs ffmpeg and other system dependencies needed for audio processing

echo "📦 Installing system dependencies..."

# Update package list
apt-get update -qq

# Install ffmpeg for audio processing
echo "🎵 Installing ffmpeg for TTS audio chunk combination..."
apt-get install -y ffmpeg

# Verify installation
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg installed successfully"
    ffmpeg -version | head -1
else
    echo "❌ ffmpeg installation failed"
    exit 1
fi

echo "✅ System dependencies installation complete"
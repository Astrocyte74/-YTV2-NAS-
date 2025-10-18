# YouTube Summarizer Bot - Custom image with ffmpeg
FROM python:3.11-slim

# Install system dependencies including ffmpeg
RUN echo "🎵🎵🎵 INSTALLING FFMPEG/FFPROBE FOR MP3 DURATION EXTRACTION 🎵🎵🎵" && \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    echo "🔍🔍🔍 VERIFYING FFMPEG/FFPROBE INSTALLATION 🔍🔍🔍" && \
    which ffmpeg && ffmpeg -version | head -1 && \
    which ffprobe && ffprobe -version | head -1 && \
    echo "✅✅✅ FFMPEG/FFPROBE INSTALLATION COMPLETE ✅✅✅"

# Set working directory
WORKDIR /app

# Copy and install Python requirements first (for better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data exports logs cache downloads

# Expose port
EXPOSE 6452

# Defaults for embedded queue worker
ENV ENABLE_TTS_QUEUE_WORKER=1 \
    TTS_QUEUE_INTERVAL=30

# Entrypoint to run both the bot and the queue worker (watch mode)
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]

#!/usr/bin/env python3
"""
NAS-to-Render Sync Module
Robust upload functionality with exponential backoff for cold starts

Required environment variables:
- RENDER_DASHBOARD_URL: Full URL to Render deployment (e.g., https://ytv2-render.onrender.com)
- SYNC_SECRET: Shared secret for authentication (must match Render's SYNC_SECRET)

The sync module automatically handles:
- Render cold starts with exponential backoff retry (6 attempts: 1s, 2s, 4s, 8s, 16s, 32s)
- Network errors and connection timeouts
- Proper multipart file upload formatting
- Audio file detection and pairing with reports
"""

import os
import time
import random
import json
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def upload_to_render(report_path, audio_path=None, max_retries=6):
    """
    Upload report and audio to Render with robust retry logic for cold starts
    
    Args:
        report_path (str/Path): Path to JSON report file
        audio_path (str/Path): Optional path to MP3 audio file (auto-detected if None)
        max_retries (int): Maximum retry attempts (default 6)
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    try:
        # Get configuration from environment
        render_url = os.environ.get('RENDER_DASHBOARD_URL', '').rstrip('/')
        sync_secret = os.environ.get('SYNC_SECRET', '')
        
        if not render_url or not sync_secret:
            logger.error("RENDER_DASHBOARD_URL and SYNC_SECRET must be set")
            return False
        
        url = f"{render_url}/api/upload-report"
        
        # Derive stem from report filename for consistent naming
        report_path = Path(report_path)
        if not report_path.exists():
            logger.error(f"Report file not found: {report_path}")
            return False
            
        stem = report_path.stem
        
        # Validate JSON contains media_metadata if audio is being uploaded
        if audio_path and Path(audio_path).exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                media_metadata = report_data.get('media_metadata', {})
                has_audio = media_metadata.get('has_audio', False)
                mp3_duration = media_metadata.get('mp3_duration_seconds')
                word_count = media_metadata.get('summary_word_count', 0)
                reading_time = media_metadata.get('estimated_reading_time_minutes', 0)
                
                if not has_audio or mp3_duration is None:
                    logger.warning(f"JSON missing MP3 metadata for {stem} - will sync anyway")
                else:
                    logger.info(f"✅ Validated JSON has MP3 metadata: {mp3_duration}s duration, {word_count} words, {reading_time}m read time")
                    
            except Exception as e:
                logger.warning(f"Could not validate JSON metadata for {stem}: {e} - will sync anyway")
        
        # Auto-pair audio if not provided using stem-based matching
        if audio_path is None:
            candidate = Path('./exports') / f"{stem}.mp3"
            if candidate.exists():
                audio_path = candidate
                logger.info(f"Auto-detected audio file: {audio_path}")
        
        # Warm-up ping to wake Render from cold start
        try:
            logger.info("🔄 Warming up Render service...")
            requests.get(f"{render_url}/health", timeout=5)
        except Exception:
            pass  # Ignore warm-up failures
        
        headers = {
            'X-Sync-Secret': sync_secret,
            'X-Report-Stem': stem  # For idempotency
        }
        
        # Retry with exponential backoff
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                logger.info(f"Upload attempt {attempt + 1}/{max_retries} for {stem}")
                
                # Use context managers to properly close files
                with open(report_path, 'rb') as rf:
                    files = {'report': (f'{stem}.json', rf, 'application/json')}
                    
                    if audio_path and Path(audio_path).exists():
                        with open(audio_path, 'rb') as af:
                            audio_filename = Path(audio_path).name  # Use actual filename
                            files['audio'] = (audio_filename, af, 'audio/mpeg')
                            logger.info(f"🎵 Uploading audio file: {audio_filename}")
                            logger.info(f"🌐 POST to: {url}")
                            response = requests.post(url, files=files, headers=headers, timeout=30)
                            logger.info(f"📡 Response: {response.status_code} - {response.text[:200]}")
                    else:
                        response = requests.post(url, files=files, headers=headers, timeout=30)
                
                if response.status_code in (200, 201):
                    result = response.json()
                    idempotent = result.get('idempotent', False)
                    status_msg = "♻️  Already synced" if idempotent else "✅ Successfully uploaded"
                    logger.info(f"{status_msg}: {stem}")
                    return True
                
                # Fatal errors - don't retry
                if response.status_code in (401, 403, 400):
                    logger.error(f"Fatal upload error ({response.status_code}): {response.text[:200]}")
                    return False
                
                # 409 means already exists but different content
                if response.status_code == 409:
                    logger.warning(f"Conflict - report exists with different content: {stem}")
                    return False
                
                # Retryable errors (502, 503, 504 for cold starts)
                logger.warning(f"Upload attempt {attempt + 1} failed: {response.status_code} - {response.text[:100]}")
                
            except requests.exceptions.RequestException as e:
                # Network errors are retryable (Render cold start, connection issues)
                logger.warning(f"Upload attempt {attempt + 1} network error: {str(e)[:100]}")
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                
            # Don't sleep on the last attempt
            if attempt < max_retries - 1:
                jitter = random.uniform(0.0, 0.4)  # Fixed range
                sleep_time = backoff + jitter
                logger.info(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                backoff *= 2  # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s
        
        logger.error(f"❌ Upload failed after {max_retries} attempts: {stem}")
        return False
        
    except Exception as e:
        logger.error(f"Upload function error: {e}")
        return False

def sync_report_to_render(video_id, timestamp, reports_dir='./data/reports', exports_dir='./exports'):
    """
    Convenience function to sync a specific report and its audio to Render
    
    Args:
        video_id (str): YouTube video ID
        timestamp (str): Report timestamp (YYYYMMDD_HHMMSS format)
        reports_dir (str): Directory containing JSON reports
        exports_dir (str): Directory containing MP3 exports
    
    Returns:
        bool: True if sync succeeded
    """
    reports_dir = Path(reports_dir)
    exports_dir = Path(exports_dir)
    
    # Find report file
    report_pattern = f"{video_id}_{timestamp}.json"
    report_path = reports_dir / report_pattern
    
    # Find audio file
    audio_pattern = f"audio_{video_id}_{timestamp}.mp3"
    audio_path = exports_dir / audio_pattern
    
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return False
    
    audio_path_arg = audio_path if audio_path.exists() else None
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path} (continuing without audio)")
    
    return upload_to_render(report_path, audio_path_arg)

if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python nas_sync.py <report_file.json> [audio_file.mp3]")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    report_file = sys.argv[1]
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = upload_to_render(report_file, audio_file)
    print(f"Upload {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1)
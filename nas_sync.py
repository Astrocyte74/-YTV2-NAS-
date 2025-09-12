#!/usr/bin/env python3
"""
NAS-to-Render Sync Module
Robust API-based synchronization replacing database file uploads

Required environment variables:
- RENDER_API_URL: Full URL to Render deployment (e.g., https://ytv2-render.onrender.com)
- SYNC_SECRET: Shared secret for authentication (must match Render's SYNC_SECRET)

The sync module automatically handles:
- Direct API calls to create/update content using UPSERT logic
- Render cold starts with exponential backoff retry
- Network errors and connection timeouts
- Audio file upload with content pairing
"""

import os
import time
import random
import json
import requests
from pathlib import Path
import logging
from modules.render_api_client import RenderAPIClient, create_client_from_env

logger = logging.getLogger(__name__)

def sync_content_via_api():
    """Sync all content to Render using API calls instead of database file upload."""
    try:
        # Use environment variable compatible with new client
        render_url = os.getenv('RENDER_DASHBOARD_URL') or os.getenv('RENDER_API_URL')
        if render_url and 'RENDER_API_URL' not in os.environ:
            os.environ['RENDER_API_URL'] = render_url
            
        client = create_client_from_env()
        
        # Test connection first
        if not client.test_connection():
            logger.error("❌ Failed to connect to Render API")
            return False
            
        logger.info("🔗 Connected to Render API successfully")
        
        # Sync all content from database via API
        db_path = Path("data/ytv2_content.db")
        if not db_path.exists():
            logger.error(f"SQLite database not found: {db_path}")
            return False
            
        logger.info(f"🗄️  Syncing content from {db_path} via API...")
        stats = client.sync_content_from_database(db_path)
        
        logger.info(f"✅ Content sync completed: {stats['created']} created, {stats['updated']} updated, {stats['errors']} errors")
        
        # Sync most recent MP3 file
        exports_dir = Path("exports")
        if exports_dir.exists():
            mp3_files = sorted(exports_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]
            
            if mp3_files:
                logger.info(f"🎵 Syncing {len(mp3_files)} recent MP3 files...")
                
                for mp3_file in mp3_files:
                    try:
                        # Extract content ID from filename (assuming format: video_id_timestamp.mp3)
                        stem = mp3_file.stem
                        # Try to find matching content by stem/video_id
                        content_id = stem  # Will be resolved by API
                        
                        result = client.upload_audio_file(mp3_file, content_id)
                        logger.info(f"✅ Synced MP3: {mp3_file.name}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Error syncing {mp3_file.name}: {e}")
            else:
                logger.info("📂 No MP3 files found in exports directory")
        else:
            logger.warning("📂 Exports directory not found - skipping MP3 sync")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ API sync error: {e}")
        return False

# Legacy function name for backward compatibility
def sync_sqlite_database():
    """Legacy function - redirects to new API-based sync."""
    logger.warning("sync_sqlite_database() is deprecated, use sync_content_via_api()")
    return sync_content_via_api()

def upload_to_render(report_path, audio_path=None, max_retries=6):
    """
    Upload report and audio to Render using new API client
    
    Args:
        report_path (str/Path): Path to JSON report file
        audio_path (str/Path): Optional path to MP3 audio file (auto-detected if None)
        max_retries (int): Maximum retry attempts (default 6)
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    try:
        # Set up environment for API client
        render_url = os.environ.get('RENDER_DASHBOARD_URL') or os.environ.get('RENDER_API_URL', '')
        if render_url and 'RENDER_API_URL' not in os.environ:
            os.environ['RENDER_API_URL'] = render_url.rstrip('/')
            
        client = create_client_from_env()
        
        report_path = Path(report_path)
        if not report_path.exists():
            logger.error(f"Report file not found: {report_path}")
            return False
            
        stem = report_path.stem
        
        # Load and validate JSON report
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Convert JSON report to universal schema format expected by API
            if 'content_source' not in report_data:
                # Legacy format - needs conversion
                logger.info(f"Converting legacy report format for {stem}")
                content_data = convert_legacy_report_to_api_format(report_data)
            else:
                # Already in universal schema format
                content_data = report_data
                
        except Exception as e:
            logger.error(f"Failed to load/parse report {stem}: {e}")
            return False
        
        # Auto-pair audio if not provided using stem-based matching
        if audio_path is None:
            candidate = Path('./exports') / f"{stem}.mp3"
            if candidate.exists():
                audio_path = candidate
                logger.info(f"Auto-detected audio file: {audio_path}")
        
        # Upload content via API
        try:
            logger.info(f"📡 Uploading content via API: {stem}")
            result = client.create_or_update_content(content_data)
            
            content_id = result.get('id') or stem
            action = result.get('action', 'unknown')
            
            status_msg = "♻️  Already exists" if action == 'updated' else "✅ Successfully created"
            logger.info(f"{status_msg}: {content_id}")
            
            # Upload audio if available
            if audio_path and Path(audio_path).exists():
                logger.info(f"🎵 Uploading audio file: {Path(audio_path).name}")
                audio_result = client.upload_audio_file(Path(audio_path), content_id)
                logger.info(f"✅ Audio uploaded successfully")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ API upload failed for {stem}: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Upload function error: {e}")
        return False


def convert_legacy_report_to_api_format(legacy_report: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy JSON report format to universal schema API format."""
    
    def parse_json_field(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return []
        return []
    
    # Extract video metadata from legacy format
    video_info = legacy_report.get('video', {})
    
    # Convert to universal schema format
    content_data = {
        'content_source': 'youtube',
        'source_metadata': {
            'youtube': {
                'video_id': video_info.get('video_id', ''),
                'title': video_info.get('title', 'Untitled'),
                'channel_name': video_info.get('channel', ''),
                'published_at': video_info.get('upload_date', ''),
                'duration_seconds': video_info.get('duration', 0),
                'thumbnail_url': video_info.get('thumbnail', ''),
                'canonical_url': video_info.get('url', '')
            }
        },
        'content_analysis': {
            'language': legacy_report.get('processing', {}).get('language', 'en'),
            'category': parse_json_field(legacy_report.get('summary', {}).get('analysis', {}).get('categories', [])),
            'content_type': legacy_report.get('summary', {}).get('analysis', {}).get('content_type', ''),
            'complexity_level': legacy_report.get('summary', {}).get('analysis', {}).get('complexity', ''),
            'key_topics': parse_json_field(legacy_report.get('summary', {}).get('analysis', {}).get('topics', [])),
            'named_entities': []  # Not available in legacy format
        },
        'media_info': {
            'has_audio': bool(legacy_report.get('media_metadata', {}).get('has_audio', False)),
            'audio_duration_seconds': legacy_report.get('media_metadata', {}).get('mp3_duration_seconds', 0),
            'has_transcript': True,  # Assume true for legacy reports
            'transcript_chars': len(str(legacy_report.get('summary', {}).get('content', ''))),
            'word_count': legacy_report.get('stats', {}).get('summary_word_count', 0)
        },
        'processing_metadata': {
            'indexed_at': legacy_report.get('metadata', {}).get('generated_at', ''),
            'content_id': legacy_report.get('metadata', {}).get('report_id', '')
        }
    }
    
    # Add summary if available
    summary_content = legacy_report.get('summary', {}).get('content', '')
    if summary_content:
        content_data['summary'] = {
            'content': {
                'summary': summary_content,
                'summary_type': legacy_report.get('summary', {}).get('type', 'comprehensive')
            }
        }
    
    return content_data

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
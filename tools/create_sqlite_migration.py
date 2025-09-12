#!/usr/bin/env python3
"""
SQLite Migration Pipeline
Converts JSON data to SQLite database with proper schema and data enrichment.
"""

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import re
import sys
from urllib.parse import urlparse, parse_qs

# Schema definition for the database
SCHEMA_SQL = """
-- Main content table
CREATE TABLE IF NOT EXISTS content (
    id TEXT PRIMARY KEY,
    
    -- Basic metadata
    title TEXT NOT NULL,
    canonical_url TEXT,
    thumbnail_url TEXT,
    published_at TEXT,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Content details
    duration_seconds INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    
    -- Media information
    has_audio BOOLEAN DEFAULT FALSE,
    audio_duration_seconds INTEGER DEFAULT 0,
    has_transcript BOOLEAN DEFAULT FALSE,
    transcript_chars INTEGER DEFAULT 0,
    
    -- YouTube specific
    video_id TEXT,
    channel_name TEXT,
    channel_id TEXT,
    view_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    
    -- Content classification
    category TEXT,  -- JSON array as text
    content_type TEXT,
    complexity_level TEXT,
    language TEXT DEFAULT 'en',
    
    -- Topics and analysis
    key_topics TEXT,  -- JSON array as text
    named_entities TEXT,  -- JSON array as text
    
    -- Processing metadata
    format_source TEXT,  -- 'complete', 'empty', 'enriched'
    processing_status TEXT DEFAULT 'imported',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_content_published_at ON content(published_at);
CREATE INDEX IF NOT EXISTS idx_content_indexed_at ON content(indexed_at);
CREATE INDEX IF NOT EXISTS idx_content_duration ON content(duration_seconds);
CREATE INDEX IF NOT EXISTS idx_content_title ON content(title);
CREATE INDEX IF NOT EXISTS idx_content_video_id ON content(video_id);
CREATE INDEX IF NOT EXISTS idx_content_channel ON content(channel_name);
CREATE INDEX IF NOT EXISTS idx_content_category ON content(category);
CREATE INDEX IF NOT EXISTS idx_content_content_type ON content(content_type);
CREATE INDEX IF NOT EXISTS idx_content_language ON content(language);

-- Content summary table for full text search
CREATE TABLE IF NOT EXISTS content_summaries (
    content_id TEXT PRIMARY KEY,
    summary_text TEXT,
    summary_type TEXT,  -- 'audio', 'video', 'text'
    FOREIGN KEY (content_id) REFERENCES content(id)
);

-- File paths table to track source files and exports
CREATE TABLE IF NOT EXISTS content_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id TEXT,
    file_type TEXT,  -- 'json', 'mp3', 'transcript'
    file_path TEXT,
    file_size INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (content_id) REFERENCES content(id)
);

-- Migration log to track what was processed
CREATE TABLE IF NOT EXISTS migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    content_id TEXT,
    status TEXT,  -- 'success', 'error', 'skipped'
    message TEXT,
    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def create_database(db_path):
    """Create SQLite database with proper schema."""
    print(f"📁 Creating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    
    print("✅ Database schema created successfully")

def extract_video_id_from_filename(filename):
    """Extract YouTube video ID from filename."""
    # Common patterns: video_title_VIDEO_ID.json or video_title_VIDEO_ID_timestamp.json
    # YouTube video IDs are 11 characters: letters, numbers, - and _
    patterns = [
        r'([a-zA-Z0-9_-]{11})\.json$',  # ends with video_id.json
        r'([a-zA-Z0-9_-]{11})_\d+\.json$',  # ends with video_id_timestamp.json
        r'_([a-zA-Z0-9_-]{11})\.json$',  # _video_id.json
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return None

def extract_title_from_filename(filename, video_id=None):
    """Extract title from filename by removing video ID and timestamp."""
    name = Path(filename).stem
    
    # Remove video ID if found
    if video_id:
        name = name.replace(video_id, '')
    
    # Remove timestamp pattern (YYYYMMDD_HHMMSS)
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    
    # Clean up underscores and make title-case
    name = name.strip('_').replace('_', ' ')
    return name.title() if name else 'Untitled'

def generate_content_id(video_id=None, title=None, url=None):
    """Generate a consistent content ID."""
    if video_id:
        return f"yt:{video_id.lower()}"
    
    # Fallback: hash of title or URL
    source = title or url or str(datetime.now())
    hash_obj = hashlib.md5(source.encode())
    return f"hash:{hash_obj.hexdigest()[:12]}"

def normalize_datetime(date_str):
    """Normalize datetime string to ISO format."""
    if not date_str:
        return None
    
    try:
        # Parse various datetime formats
        if date_str.endswith('Z'):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        
        return dt.isoformat()
    except:
        return None

def process_json_file(file_path, conn):
    """Process a single JSON file and extract data for database."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine format type
        has_title = bool(data.get('title', '').strip())
        has_url = bool(data.get('canonical_url', '').strip())
        
        if has_title and has_url:
            format_source = 'complete'
        else:
            format_source = 'empty'
        
        # Extract video ID from multiple sources
        video_id = None
        if 'source_metadata' in data and 'youtube' in data['source_metadata']:
            video_id = data['source_metadata']['youtube'].get('video_id')
        
        if not video_id:
            video_id = extract_video_id_from_filename(file_path.name)
        
        # Generate title if missing
        title = data.get('title', '').strip()
        if not title:
            title = extract_title_from_filename(file_path.name, video_id)
        
        # Generate URL if missing
        canonical_url = data.get('canonical_url', '').strip()
        if not canonical_url and video_id:
            canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Generate thumbnail URL if missing  
        thumbnail_url = data.get('thumbnail_url', '').strip()
        if not thumbnail_url and video_id:
            thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hq720.jpg"
        
        # Generate content ID
        content_id = generate_content_id(video_id, title, canonical_url)
        
        # Extract YouTube metadata
        yt_meta = data.get('source_metadata', {}).get('youtube', {})
        channel_name = yt_meta.get('channel_name', 'Unknown')
        if channel_name == 'Unknown' and yt_meta.get('uploader_id'):
            channel_name = yt_meta.get('uploader_id', '').replace('@', '')
        
        # Extract analysis data
        analysis = data.get('analysis', {})
        
        # Prepare content record
        content_record = {
            'id': content_id,
            'title': title,
            'canonical_url': canonical_url,
            'thumbnail_url': thumbnail_url,
            'published_at': normalize_datetime(data.get('published_at')),
            'indexed_at': normalize_datetime(data.get('indexed_at')) or datetime.now(timezone.utc).isoformat(),
            'duration_seconds': int(data.get('duration_seconds', 0)),
            'word_count': int(data.get('word_count', 0)),
            'has_audio': bool(data.get('media', {}).get('has_audio', False)),
            'audio_duration_seconds': int(data.get('media', {}).get('audio_duration_seconds', 0)),
            'has_transcript': bool(data.get('media', {}).get('has_transcript', False)),
            'transcript_chars': int(data.get('media', {}).get('transcript_chars', 0)),
            'video_id': video_id,
            'channel_name': channel_name,
            'channel_id': yt_meta.get('channel_id', ''),
            'view_count': int(yt_meta.get('view_count', 0)),
            'like_count': int(yt_meta.get('like_count', 0)),
            'comment_count': int(yt_meta.get('comment_count', 0)),
            'category': json.dumps(analysis.get('category', [])),
            'content_type': analysis.get('content_type', ''),
            'complexity_level': analysis.get('complexity_level', ''),
            'language': analysis.get('language', 'en'),
            'key_topics': json.dumps(analysis.get('key_topics', [])),
            'named_entities': json.dumps(analysis.get('named_entities', [])),
            'format_source': format_source
        }
        
        # Insert content record
        placeholders = ', '.join(['?' for _ in content_record])
        columns = ', '.join(content_record.keys())
        
        conn.execute(
            f"INSERT OR REPLACE INTO content ({columns}) VALUES ({placeholders})",
            list(content_record.values())
        )
        
        # Extract and store summary if available
        summary_data = data.get('summary', {})
        if summary_data and 'content' in summary_data:
            summary_content = summary_data['content']
            summary_text = summary_content.get('summary', '')
            summary_type = summary_content.get('summary_type', 'unknown')
            
            if summary_text.strip():
                conn.execute(
                    "INSERT OR REPLACE INTO content_summaries (content_id, summary_text, summary_type) VALUES (?, ?, ?)",
                    (content_id, summary_text, summary_type)
                )
        
        # Record file path
        conn.execute(
            "INSERT INTO content_files (content_id, file_type, file_path, file_size) VALUES (?, ?, ?, ?)",
            (content_id, 'json', str(file_path), file_path.stat().st_size)
        )
        
        # Log successful processing
        conn.execute(
            "INSERT INTO migration_log (source_file, content_id, status, message) VALUES (?, ?, ?, ?)",
            (file_path.name, content_id, 'success', f'Processed as {format_source} format')
        )
        
        return content_id, format_source
        
    except Exception as e:
        # Log error
        conn.execute(
            "INSERT INTO migration_log (source_file, content_id, status, message) VALUES (?, ?, ?, ?)",
            (file_path.name, None, 'error', str(e))
        )
        raise

def migrate_json_to_sqlite(json_dir, db_path):
    """Migrate all JSON files to SQLite database."""
    json_dir = Path(json_dir)
    
    if not json_dir.exists():
        print(f"❌ JSON directory not found: {json_dir}")
        return False
    
    # Create database
    create_database(db_path)
    
    # Find JSON files
    json_files = list(json_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith('.')]
    
    if not json_files:
        print(f"❌ No JSON files found in {json_dir}")
        return False
    
    print(f"🔄 Migrating {len(json_files)} JSON files to SQLite...")
    
    # Process files
    conn = sqlite3.connect(db_path)
    successful = 0
    errors = 0
    format_counts = {'complete': 0, 'empty': 0}
    
    try:
        for i, json_file in enumerate(json_files, 1):
            try:
                content_id, format_source = process_json_file(json_file, conn)
                format_counts[format_source] += 1
                successful += 1
                
                if i % 10 == 0 or i == len(json_files):
                    print(f"Progress: {i}/{len(json_files)} files processed ({successful} success, {errors} errors)")
                    
            except Exception as e:
                errors += 1
                print(f"❌ Error processing {json_file.name}: {e}")
        
        conn.commit()
        
    finally:
        conn.close()
    
    # Final report
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully migrated: {successful} files")
    print(f"❌ Errors encountered: {errors} files")
    print(f"📁 Complete format files: {format_counts['complete']}")
    print(f"📝 Empty format files: {format_counts['empty']}")
    print(f"💾 Database created: {db_path}")
    print()
    
    return True

def verify_database(db_path):
    """Verify the migrated database."""
    print(f"🔍 Verifying database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Basic counts
        cursor = conn.execute("SELECT COUNT(*) FROM content")
        total_content = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM content_summaries")
        total_summaries = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM content_files")
        total_files = cursor.fetchone()[0]
        
        # Data quality checks
        cursor = conn.execute("SELECT COUNT(*) FROM content WHERE title != ''")
        with_titles = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM content WHERE canonical_url != ''")
        with_urls = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM content WHERE duration_seconds > 0")
        with_durations = cursor.fetchone()[0]
        
        # Format breakdown
        cursor = conn.execute("SELECT format_source, COUNT(*) FROM content GROUP BY format_source")
        format_breakdown = dict(cursor.fetchall())
        
        print("📊 DATABASE VERIFICATION")
        print(f"   Total content records: {total_content}")
        print(f"   Content summaries: {total_summaries}")
        print(f"   File references: {total_files}")
        print()
        print("📈 DATA QUALITY")
        print(f"   Records with titles: {with_titles}/{total_content} ({with_titles/total_content*100:.1f}%)")
        print(f"   Records with URLs: {with_urls}/{total_content} ({with_urls/total_content*100:.1f}%)")  
        print(f"   Records with durations: {with_durations}/{total_content} ({with_durations/total_content*100:.1f}%)")
        print()
        print("📁 FORMAT BREAKDOWN")
        for format_type, count in format_breakdown.items():
            print(f"   {format_type}: {count} records")
        
    finally:
        conn.close()
    
    print("✅ Database verification complete")

def main():
    # Configuration
    json_dir = "/Users/markdarby/projects/YTV_temp_NAS_files/data/reports"
    db_path = "data/ytv2_content.db"
    
    print("🚀 YTV2 SQLite Migration Pipeline")
    print("=" * 50)
    print()
    
    # Run migration
    success = migrate_json_to_sqlite(json_dir, db_path)
    
    if success:
        verify_database(db_path)
        
        print()
        print("🎉 Migration pipeline completed successfully!")
        print(f"📄 Next steps:")
        print(f"   1. Test database queries")
        print(f"   2. Update dashboard backend to use SQLite")
        print(f"   3. Backup original JSON files")
        print(f"   4. Deploy updated dashboard")
        
        return 0
    else:
        print("❌ Migration failed")
        return 1

if __name__ == "__main__":
    exit(main())
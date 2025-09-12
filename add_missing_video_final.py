#!/usr/bin/env python3
"""
Add the missing JWST video to SQLite database (final working version)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys

def add_video_to_database():
    """Add the JWST video to SQLite database from its JSON file"""
    
    # Find the JSON file
    json_file = Path('data/reports/jwst_broke_physics_new_discoveries_challenge_cosmo_wFGUMbXwdwY.json')
    
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return False
        
    db_path = Path('data/ytv2_content.db')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
        
    try:
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"📄 Loaded JSON data for: {data.get('title', 'Unknown')}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the ID - it's already in the format "yt:wfgumbxwdwy"
        video_id = data.get('id', '')
        if not video_id:
            print("❌ No id found in JSON")
            return False
            
        print(f"🎯 Video ID: {video_id}")
        
        # Check if already exists
        cursor.execute('SELECT id FROM content WHERE id = ?', (video_id,))
        existing = cursor.fetchone()
        
        if existing:
            print(f"✅ Video already exists in database: {video_id}")
            conn.close()
            return True
            
        # Extract video ID from the full id (remove "yt:" prefix)
        short_video_id = video_id.replace('yt:', '') if video_id.startswith('yt:') else video_id
            
        # Extract channel from metadata
        metadata = data.get('metadata', {})
        channel_name = (metadata.get('uploader') or 
                       metadata.get('channel') or 
                       metadata.get('uploader_id') or 
                       'Unknown')
        
        # Extract analysis data
        analysis = data.get('analysis', {})
        
        # Get current timestamp
        now = datetime.now().isoformat()
        
        # Prepare record data matching the exact schema
        record_data = {
            'id': video_id,
            'title': data.get('title', 'Untitled'),
            'canonical_url': data.get('canonical_url', ''),
            'thumbnail_url': data.get('thumbnail_url', ''),
            'published_at': data.get('published_at', ''),
            'indexed_at': data.get('processed_at', now),
            'duration_seconds': data.get('duration_seconds', 0),
            'word_count': data.get('word_count', 0),
            'has_audio': bool(data.get('media_metadata', {}).get('mp3_file')),
            'audio_duration_seconds': data.get('media_metadata', {}).get('duration_seconds'),
            'has_transcript': bool(data.get('transcript')),
            'transcript_chars': len(data.get('transcript', '')),
            'video_id': short_video_id,
            'channel_name': channel_name,
            'channel_id': metadata.get('channel_id', ''),
            'view_count': metadata.get('view_count'),
            'like_count': metadata.get('like_count'),
            'comment_count': metadata.get('comment_count'),
            'category': ', '.join(analysis.get('category', [])) if analysis.get('category') else None,
            'content_type': analysis.get('content_type'),
            'complexity_level': analysis.get('complexity_level'),
            'language': analysis.get('language'),
            'key_topics': ', '.join(analysis.get('key_topics', [])) if analysis.get('key_topics') else None,
            'named_entities': ', '.join(analysis.get('named_entities', [])) if analysis.get('named_entities') else None,
            'format_source': 'json_migration',
            'processing_status': 'completed',
            'created_at': now,
            'updated_at': now
        }
        
        print(f"📊 Preparing to insert:")
        print(f"   Title: {record_data['title']}")
        print(f"   Duration: {record_data['duration_seconds']}s")
        print(f"   Channel: {record_data['channel_name']}")
        print(f"   Has Audio: {record_data['has_audio']}")
        
        # Insert into database using exact column names
        insert_sql = '''
        INSERT OR REPLACE INTO content 
        (id, title, canonical_url, thumbnail_url, published_at, indexed_at, duration_seconds,
         word_count, has_audio, audio_duration_seconds, has_transcript, transcript_chars,
         video_id, channel_name, channel_id, view_count, like_count, comment_count,
         category, content_type, complexity_level, language, key_topics, named_entities,
         format_source, processing_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        cursor.execute(insert_sql, (
            record_data['id'],
            record_data['title'],
            record_data['canonical_url'],
            record_data['thumbnail_url'],
            record_data['published_at'],
            record_data['indexed_at'],
            record_data['duration_seconds'],
            record_data['word_count'],
            record_data['has_audio'],
            record_data['audio_duration_seconds'],
            record_data['has_transcript'],
            record_data['transcript_chars'],
            record_data['video_id'],
            record_data['channel_name'],
            record_data['channel_id'],
            record_data['view_count'],
            record_data['like_count'],
            record_data['comment_count'],
            record_data['category'],
            record_data['content_type'],
            record_data['complexity_level'],
            record_data['language'],
            record_data['key_topics'],
            record_data['named_entities'],
            record_data['format_source'],
            record_data['processing_status'],
            record_data['created_at'],
            record_data['updated_at']
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully added video to database!")
        return True
        
    except Exception as e:
        print(f"❌ Error adding video to database: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 Adding missing JWST video to SQLite database...")
    
    success = add_video_to_database()
    
    if success:
        # Check the count and verify the video
        conn = sqlite3.connect('data/ytv2_content.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM content')
        count = cursor.fetchone()[0]
        
        # Verify the video was added
        cursor.execute('SELECT title, channel_name, duration_seconds FROM content WHERE id LIKE "%wfgumbx%" OR title LIKE "%JWST%"')
        jwst_video = cursor.fetchone()
        
        conn.close()
        
        print(f"📊 Database now has {count} records")
        if jwst_video:
            print(f"✅ JWST video confirmed in database:")
            print(f"   Title: {jwst_video[0]}")
            print(f"   Channel: {jwst_video[1]}")
            print(f"   Duration: {jwst_video[2]}s")
            print("🔄 Ready to sync to Render!")
        else:
            print("❌ JWST video not found after insertion")
    else:
        print("❌ Failed to add video")
        sys.exit(1)

if __name__ == "__main__":
    main()
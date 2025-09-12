#!/usr/bin/env python3
"""
Check the newest videos in the SQLite database
"""

import sqlite3
from pathlib import Path

def main():
    db_path = Path('data/ytv2_content.db')
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM content')
        total = cursor.fetchone()[0]
        print(f'📊 Total records in NAS database: {total}')
        print()
        
        cursor.execute('SELECT id, title, indexed_at FROM content ORDER BY indexed_at DESC LIMIT 5')
        recent = cursor.fetchall()
        
        print('Most recent 5 videos in NAS database:')
        for i, video in enumerate(recent, 1):
            video_id = video[0]
            title = video[1][:60] + ('...' if len(video[1]) > 60 else '')
            date = video[2]
            print(f'{i}. {video_id}')
            print(f'   Title: {title}')
            print(f'   Date: {date}')
            
            # Check if this looks like the Vermont video
            if 'vermont' in title.lower() or 'CEF_ya1tLfU' in video_id.lower() or 'ya1tlfu' in video_id.lower():
                print(f'   ✅ This looks like the new Vermont video!')
            print()
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    main()
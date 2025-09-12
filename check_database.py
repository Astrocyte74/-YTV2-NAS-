#!/usr/bin/env python3
"""
Check SQLite database contents for new video
"""

import sqlite3
from pathlib import Path

def main():
    db_path = Path('data/ytv2_content.db')
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
        
    print(f"📊 Checking database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check total count
        cursor.execute('SELECT COUNT(*) as total FROM content')
        total = cursor.fetchone()[0]
        print(f'Total records in database: {total}')
        
        # Check for JWST video specifically
        cursor.execute('''
            SELECT id, title, indexed_at 
            FROM content 
            WHERE title LIKE "%JWST%" OR title LIKE "%jwst%" OR id LIKE "%wFGUMbX%" 
            ORDER BY indexed_at DESC 
            LIMIT 5
        ''')
        jwst_videos = cursor.fetchall()
        
        print('\n🔍 JWST videos found:')
        if jwst_videos:
            for video in jwst_videos:
                print(f'  ✅ ID: {video[0]}')
                print(f'     Title: {video[1]}')
                print(f'     Date: {video[2]}')
                print()
        else:
            print('  ❌ No JWST videos found in database')
            
            print('\n📋 Most recent 5 videos in database:')
            cursor.execute('SELECT id, title, indexed_at FROM content ORDER BY indexed_at DESC LIMIT 5')
            recent = cursor.fetchall()
            for i, video in enumerate(recent, 1):
                title = video[1] if len(video[1]) <= 60 else video[1][:60] + '...'
                print(f'  {i}. {video[0]} - {title} ({video[2]})')
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    main()
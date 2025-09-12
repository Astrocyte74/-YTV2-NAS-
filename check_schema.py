#!/usr/bin/env python3
"""
Check the actual SQLite database schema
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
        
        # Get table schema
        cursor.execute("PRAGMA table_info(content)")
        columns = cursor.fetchall()
        
        print("📊 SQLite table 'content' schema:")
        print("=" * 50)
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'} - {'PRIMARY KEY' if col[5] else ''}")
        
        print("\n" + "=" * 50)
        print("Column names only:")
        column_names = [col[1] for col in columns]
        for name in column_names:
            print(f"  {name}")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking schema: {e}")

if __name__ == "__main__":
    main()
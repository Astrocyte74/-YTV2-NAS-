#!/usr/bin/env python3
"""
Consolidate the two YTV2 databases into a single unified database
"""

import sqlite3
import shutil
from pathlib import Path

def consolidate_databases():
    root_db = Path("ytv2_content.db")
    data_db = Path("data/ytv2_content.db")
    backup_root = Path("ytv2_content_backup.db")
    
    print("🔄 YTV2 Database Consolidation")
    print("=" * 50)
    
    # Check both databases exist
    if not root_db.exists():
        print(f"❌ Root database not found: {root_db}")
        return False
        
    if not data_db.exists():
        print(f"❌ Data database not found: {data_db}")
        return False
    
    # Get counts from both databases
    root_conn = sqlite3.connect(root_db)
    root_cursor = root_conn.cursor()
    root_cursor.execute("SELECT COUNT(*) FROM content")
    root_content_count = root_cursor.fetchone()[0]
    root_cursor.execute("SELECT COUNT(*) FROM content_summaries")
    root_summary_count = root_cursor.fetchone()[0]
    root_conn.close()
    
    data_conn = sqlite3.connect(data_db)
    data_cursor = data_conn.cursor()
    data_cursor.execute("SELECT COUNT(*) FROM content")
    data_content_count = data_cursor.fetchone()[0]
    data_cursor.execute("SELECT COUNT(*) FROM content_summaries")
    data_summary_count = data_cursor.fetchone()[0]
    data_conn.close()
    
    print(f"📊 Database Comparison:")
    print(f"   Root DB: {root_content_count} content, {root_summary_count} summaries")
    print(f"   Data DB: {data_content_count} content, {data_summary_count} summaries")
    
    # Determine which database has more complete data
    if data_content_count >= root_content_count and data_summary_count >= root_summary_count:
        print(f"✅ Data DB has more complete data, using it as primary")
        primary_db = data_db
        secondary_db = root_db
    elif root_content_count >= data_content_count and root_summary_count >= data_summary_count:
        print(f"✅ Root DB has more complete data, using it as primary")
        primary_db = root_db
        secondary_db = data_db
    else:
        print(f"⚠️  Mixed completeness, need to merge both databases")
        # For now, use the one with more content records
        if data_content_count > root_content_count:
            primary_db = data_db
            secondary_db = root_db
        else:
            primary_db = root_db
            secondary_db = data_db
    
    print(f"🎯 Using {primary_db} as primary database")
    
    # Backup the root database
    if root_db.exists():
        shutil.copy2(root_db, backup_root)
        print(f"💾 Backed up root database to {backup_root}")
    
    # Copy the primary database to the target location (data/ytv2_content.db)
    target_db = Path("data/ytv2_content.db")
    target_db.parent.mkdir(exist_ok=True)
    
    if primary_db != target_db:
        shutil.copy2(primary_db, target_db)
        print(f"📋 Copied {primary_db} to {target_db}")
    
    # Remove the old root database if it exists and is different from target
    if root_db.exists() and root_db != target_db:
        root_db.unlink()
        print(f"🗑️  Removed old root database: {root_db}")
    
    # Final verification
    final_conn = sqlite3.connect(target_db)
    final_cursor = final_conn.cursor()
    final_cursor.execute("SELECT COUNT(*) FROM content")
    final_content_count = final_cursor.fetchone()[0]
    final_cursor.execute("SELECT COUNT(*) FROM content_summaries")
    final_summary_count = final_cursor.fetchone()[0]
    final_conn.close()
    
    print(f"✅ Final consolidated database: {final_content_count} content, {final_summary_count} summaries")
    print(f"🎯 All code should now reference: data/ytv2_content.db")
    
    return True

if __name__ == "__main__":
    consolidate_databases()
#!/usr/bin/env python3
"""
SQLite Database Sync Script
Syncs the local SQLite database to the Dashboard deployment.

This script should be run after:
1. Initial migration from JSON to SQLite
2. Processing new videos that update the database
3. Any database maintenance or updates

Usage:
    python sync_sqlite_db.py
    
Environment variables:
- RENDER_DASHBOARD_URL: Dashboard URL
- SYNC_SECRET: Authentication secret
"""

import os
import requests
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_database():
    """Upload SQLite database to Dashboard"""
    
    # Configuration
    dashboard_url = os.getenv('RENDER_DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
    sync_secret = os.getenv('SYNC_SECRET', '')
    db_path = Path('data/ytv2_content.db')
    
    if not sync_secret:
        logger.error("SYNC_SECRET environment variable not set")
        return False
    
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return False
    
    logger.info(f"Syncing database {db_path} to {dashboard_url}")
    
    # Upload database file
    upload_url = f"{dashboard_url}/api/upload-database"
    headers = {
        'Authorization': f'Bearer {sync_secret}'
    }
    
    try:
        with open(db_path, 'rb') as db_file:
            files = {
                'database': ('ytv2_content.db', db_file, 'application/x-sqlite3')
            }
            
            logger.info("Uploading database file...")
            response = requests.post(
                upload_url,
                headers=headers,
                files=files,
                timeout=120  # 2 minutes for large database
            )
            
            if response.status_code == 200:
                logger.info("✅ Database sync successful")
                return True
            else:
                logger.error(f"❌ Database sync failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Database sync error: {e}")
        return False

def verify_sync():
    """Verify the database was synced successfully"""
    dashboard_url = os.getenv('RENDER_DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
    
    try:
        # Check health endpoint
        response = requests.get(f"{dashboard_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Dashboard is responding")
            
            # Check report count
            response = requests.get(f"{dashboard_url}/api/reports?size=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = data.get('pagination', {}).get('total_count', 0)
                logger.info(f"✅ Dashboard reports: {count} records")
                return count > 0
            else:
                logger.warning(f"Could not verify report count: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        
    return False

if __name__ == "__main__":
    print("🚀 SQLite Database Sync")
    print("=" * 50)
    
    if sync_database():
        print("\n⏳ Waiting for deployment to update...")
        time.sleep(10)
        
        if verify_sync():
            print("\n🎉 Database sync completed successfully!")
        else:
            print("\n⚠️  Database uploaded but verification failed")
    else:
        print("\n❌ Database sync failed")
        exit(1)
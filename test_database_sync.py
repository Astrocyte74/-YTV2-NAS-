#!/usr/bin/env python3
"""
Test Database Sync to Render
This will sync the modified database with the test title change to verify SQLite is being used.
"""

import os
import logging
from pathlib import Path
from nas_sync import sync_sqlite_database

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test database sync with modified title"""
    
    # Check required environment variables
    render_url = os.getenv('RENDER_DASHBOARD_URL')
    sync_secret = os.getenv('SYNC_SECRET')
    
    if not render_url or not sync_secret:
        logger.error("❌ Missing RENDER_DASHBOARD_URL or SYNC_SECRET environment variables")
        logger.error("Make sure you're running this from inside the Docker container")
        return False
    
    logger.info("🧪 Testing database sync with modified title...")
    logger.info("📝 Updated AirPods title to: '🎧 TESTING: Apple AirPods 3 Database Sync Verification 🎧'")
    logger.info(f"📡 Syncing to: {render_url}")
    
    # Sync the database
    success = sync_sqlite_database()
    
    if success:
        logger.info("✅ Database synced successfully!")
        logger.info("🔍 Check the dashboard - if you see the new title, SQLite is working!")
        logger.info("📋 Look for: '🎧 TESTING: Apple AirPods 3 Database Sync Verification 🎧'")
        logger.info("🌐 Dashboard URL: https://ytv2-vy9k.onrender.com/")
        return True
    else:
        logger.error("❌ Database sync failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
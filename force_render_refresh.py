#!/usr/bin/env python3
"""
Force Render Dashboard Refresh
This script will force the Render dashboard to refresh its content index
by re-uploading the database, which should trigger it to use the latest code.
"""

import os
import logging
from pathlib import Path
from nas_sync import sync_sqlite_database

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Force refresh the Render dashboard"""
    
    # Check required environment variables
    render_url = os.getenv('RENDER_DASHBOARD_URL')
    sync_secret = os.getenv('SYNC_SECRET')
    
    if not render_url or not sync_secret:
        logger.error("❌ Missing RENDER_DASHBOARD_URL or SYNC_SECRET environment variables")
        logger.error("Make sure you're running this from inside the Docker container")
        return False
    
    logger.info("🔄 Force refreshing Render dashboard...")
    logger.info(f"📡 Target URL: {render_url}")
    
    # Re-sync the database to trigger refresh
    success = sync_sqlite_database()
    
    if success:
        logger.info("✅ Database re-synced successfully!")
        logger.info("🎯 This should force Render to reload with the latest sorting fixes")
        logger.info("📝 Try the dashboard sorting now - it should work!")
        return True
    else:
        logger.error("❌ Database sync failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
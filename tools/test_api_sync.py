#!/usr/bin/env python3
"""
Test script for new API-based sync functionality.
Validates that the API client can connect and sync content.
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root and modules directory are on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = ROOT_DIR / 'modules'

for path in (ROOT_DIR, MODULES_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from modules.render_api_client import create_client_from_env
from nas_sync import sync_content_via_api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_api_connection():
    """Test basic API connectivity."""
    print("🔍 Testing API connection...")
    
    try:
        # Check environment variables
        render_url = os.getenv('RENDER_API_URL') or os.getenv('RENDER_DASHBOARD_URL')
        sync_secret = os.getenv('SYNC_SECRET')
        
        if not render_url:
            print("❌ RENDER_API_URL or RENDER_DASHBOARD_URL not set")
            return False
            
        if not sync_secret:
            print("❌ SYNC_SECRET not set")
            return False
            
        print(f"✅ Environment configured: {render_url}")
        
        # Create API client
        if 'RENDER_API_URL' not in os.environ and render_url:
            os.environ['RENDER_API_URL'] = render_url
            
        client = create_client_from_env()
        
        # Test connection
        if client.test_connection():
            print("✅ API connection successful!")
            return True
        else:
            print("❌ API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def test_database_sync():
    """Test syncing content from database via API."""
    print("\n🔍 Testing database sync via API...")
    
    # Check if database exists
    db_path = Path("data/ytv2_content.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
        
    print(f"✅ Database found: {db_path}")
    
    try:
        success = sync_content_via_api()
        
        if success:
            print("✅ Database sync via API successful!")
            return True
        else:
            print("❌ Database sync via API failed")
            return False
            
    except Exception as e:
        print(f"❌ Database sync error: {e}")
        return False

def test_legacy_upload():
    """Test uploading a single report file using legacy function."""
    print("\n🔍 Testing legacy upload function...")
    
    # Look for a sample report file
    reports_dir = Path("data/reports")
    if not reports_dir.exists():
        print("❌ No reports directory found")
        return False
        
    json_files = list(reports_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON report files found")
        return False
        
    sample_report = json_files[0]
    print(f"📄 Using sample report: {sample_report}")
    
    try:
        from nas_sync import upload_to_render
        success = upload_to_render(sample_report)
        
        if success:
            print("✅ Legacy upload successful!")
            return True
        else:
            print("❌ Legacy upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Legacy upload error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing new API-based sync functionality\n")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: API Connection
    total_tests += 1
    if test_api_connection():
        tests_passed += 1
        
    # Test 2: Database sync (only if connection worked)
    total_tests += 1
    if test_database_sync():
        tests_passed += 1
        
    # Test 3: Legacy upload compatibility
    total_tests += 1
    if test_legacy_upload():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API sync is ready for production.")
        return True
    else:
        print("⚠️  Some tests failed. Check configuration and connectivity.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

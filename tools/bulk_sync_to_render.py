#!/usr/bin/env python3
"""
Bulk Sync Backfilled JSON Files to Render Dashboard
Uploads all JSON files with media_metadata from NAS to Render using the existing API
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict

class RenderBulkSync:
    def __init__(self, reports_dir: str, render_url: str):
        self.reports_dir = Path(reports_dir)
        self.render_url = render_url.rstrip('/')
        self.upload_endpoint = f"{self.render_url}/api/upload-report"
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'enhanced_files': 0,
            'uploaded': 0,
            'skipped': 0,
            'errors': 0,
        }
        
    def log(self, message: str):
        """Log with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def is_enhanced_file(self, file_path: Path) -> bool:
        """Check if JSON file has media metadata from backfill"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check for backfilled media_metadata fields
            media_metadata = data.get('media_metadata', {})
            return bool(media_metadata)  # Return True if media_metadata exists and is not empty
            
        except Exception as e:
            self.log(f"❌ Error checking {file_path.name}: {e}")
            return False
    
    def upload_file_to_render(self, file_path: Path) -> bool:
        """Upload a single JSON file to Render"""
        try:
            report_stem = file_path.stem
            
            # Get sync secret from environment or use hardcoded value
            sync_secret = os.environ.get('SYNC_SECRET', '5397064f171ce0db328066d2ac52022b')
            if not sync_secret:
                self.log(f"❌ SYNC_SECRET not available")
                return False
            
            # Headers for authentication
            headers = {
                'X-Sync-Secret': sync_secret,
                'X-Report-Stem': report_stem
            }
            
            # Read the JSON file
            with open(file_path, 'rb') as f:
                files = {
                    'report': (file_path.name, f, 'application/json')
                }
                
                # Upload to Render API
                self.log(f"📤 Uploading {file_path.name}...")
                response = requests.post(
                    self.upload_endpoint,
                    files=files,
                    headers=headers,
                    timeout=30
                )
            
            if response.status_code in (200, 201):
                result = response.json()
                idempotent = result.get('idempotent', False)
                status_msg = "♻️ Already synced" if idempotent else "✅ Successfully uploaded"
                self.log(f"{status_msg}: {result.get('message', 'Upload complete')}")
                return True
            elif response.status_code == 409:
                self.log(f"⚠️ File exists with different content (skipping): {file_path.name}")
                return True  # Count as success since file exists
            else:
                self.log(f"❌ Upload failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    self.log(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    self.log(f"   Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.Timeout:
            self.log(f"⏰ Upload timeout for {file_path.name}")
            return False
        except Exception as e:
            self.log(f"❌ Upload error for {file_path.name}: {e}")
            return False
    
    def get_enhanced_files(self) -> List[Path]:
        """Get list of all JSON files with media_metadata"""
        backfilled_files = []
        
        for json_file in self.reports_dir.glob('*.json'):
            if json_file.name.startswith('._'):
                continue
                
            self.stats['total_files'] += 1
            
            if self.is_enhanced_file(json_file):
                backfilled_files.append(json_file)
                self.stats['enhanced_files'] += 1
        
        return backfilled_files
    
    def warm_up_render_service(self) -> bool:
        """Warm up the Render service before bulk upload"""
        try:
            self.log("🔥 Warming up Render service...")
            response = requests.get(self.render_url, timeout=30)
            if response.status_code == 200:
                self.log("✅ Render service is ready")
                return True
            else:
                self.log(f"⚠️ Render service responded with {response.status_code}")
                return True  # Continue anyway
        except Exception as e:
            self.log(f"⚠️ Could not warm up service: {e}")
            return True  # Continue anyway
    
    def run_bulk_sync(self, batch_size: int = 5, delay_between_batches: int = 2):
        """Run the bulk sync process"""
        self.log("🚀 Starting Render Bulk Sync")
        self.log(f"📁 Reports directory: {self.reports_dir}")
        self.log(f"🎯 Target URL: {self.render_url}")
        
        # Warm up service
        if not self.warm_up_render_service():
            self.log("❌ Service warm-up failed, aborting")
            return False
        
        # Get backfilled files
        self.log("🔍 Scanning for backfilled JSON files with media_metadata...")
        enhanced_files = self.get_enhanced_files()
        
        if not enhanced_files:
            self.log("❌ No backfilled files with media_metadata found!")
            return False
        
        self.log(f"📊 Found {len(enhanced_files)} backfilled files out of {self.stats['total_files']} total files")
        
        # Upload in batches
        batch_count = 0
        for i in range(0, len(enhanced_files), batch_size):
            batch = enhanced_files[i:i + batch_size]
            batch_count += 1
            
            self.log(f"\n📦 Batch {batch_count}: Uploading {len(batch)} files...")
            
            for file_path in batch:
                if self.upload_file_to_render(file_path):
                    self.stats['uploaded'] += 1
                else:
                    self.stats['errors'] += 1
                
                # Small delay between individual uploads
                time.sleep(0.5)
            
            # Delay between batches
            if i + batch_size < len(enhanced_files):
                self.log(f"⏸️ Batch {batch_count} complete. Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("BULK SYNC SUMMARY")
        self.log("="*60)
        self.log(f"📊 Total files scanned: {self.stats['total_files']}")
        self.log(f"🔧 Backfilled files found: {self.stats['enhanced_files']}")
        self.log(f"✅ Successfully uploaded: {self.stats['uploaded']}")
        self.log(f"❌ Upload errors: {self.stats['errors']}")
        
        success_rate = (self.stats['uploaded'] / self.stats['enhanced_files'] * 100) if self.stats['enhanced_files'] > 0 else 0
        self.log(f"📈 Success rate: {success_rate:.1f}%")
        
        return self.stats['errors'] == 0


def main():
    # Configuration
    reports_dir = "/volume1/Docker/YTV2/data/reports"
    render_url = "https://ytv2-dashboard-postgres.onrender.com"
    
    # Validate directory
    if not Path(reports_dir).exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    # Initialize sync
    sync = RenderBulkSync(reports_dir, render_url)
    
    # Run bulk sync
    success = sync.run_bulk_sync(batch_size=3, delay_between_batches=3)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

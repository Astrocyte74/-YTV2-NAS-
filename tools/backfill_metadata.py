#!/usr/bin/env python3
"""
YTV2 Metadata Backfill Tool - Enhanced Robust Version
Backfills existing JSON reports with missing YouTube metadata fields
Idempotent, resumable, rate-limited with atomic writes and robust YouTube API handling
"""

import argparse
import json
import os
import sys
import time
import tempfile
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp


class MetadataBackfiller:
    def __init__(self, reports_dir: str, dry_run: bool = False, force: bool = False):
        self.reports_dir = Path(reports_dir)
        self.dry_run = dry_run
        self.force = force
        
        # State tracking for resume capability
        self.state_file = self.reports_dir / '.backfill_state'
        self.processed_files = set()
        
        # Enhanced rate limiting - smaller batches, longer sleep
        self.batch_size = 10  # Reduced from 50 for gentler approach
        self.sleep_duration = 30  # Increased from 15s
        self.request_sleep = 3  # Sleep between individual requests
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'api_calls': 0,
            'rate_limited': 0,
            'network_errors': 0,
        }
        
        # Setup logging
        log_file = f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message: str, print_also: bool = True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
        
        if print_also:
            print(log_entry)
    
    def load_state(self):
        """Load previous processing state for resume"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.processed_files = set(state.get('processed_files', []))
                self.log(f"📁 Resumed: {len(self.processed_files)} files already processed")
            except Exception as e:
                self.log(f"⚠️ Could not load state file: {e}")
                self.processed_files = set()
    
    def save_state(self, current_file: str):
        """Save current processing state"""
        self.processed_files.add(current_file)
        state = {
            'processed_files': list(self.processed_files),
            'last_updated': datetime.now().isoformat(),
            'stats': self.stats.copy(),
        }
        
        # Atomic write
        temp_file = str(self.state_file) + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        os.rename(temp_file, self.state_file)
    
    def extract_video_id_from_json(self, json_path: Path) -> Optional[str]:
        """Extract video_id from JSON file, trying multiple approaches"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Method 1: Universal schema location
            if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                video_id = data['source_metadata']['youtube'].get('video_id')
                if video_id and video_id not in ['', 'unknown', 'error']:
                    return video_id
            
            # Method 2: Legacy metadata locations
            if 'metadata' in data:
                metadata = data['metadata']
                video_id = metadata.get('video_id') or metadata.get('id')
                if video_id and video_id not in ['', 'unknown', 'error']:
                    return video_id
            
            # Method 3: Legacy video object
            if 'video' in data:
                video = data['video']
                video_id = video.get('video_id') or video.get('id')
                if video_id and video_id not in ['', 'unknown', 'error']:
                    return video_id
            
            # Method 4: Top-level locations
            video_id = data.get('video_id') or data.get('id')
            if video_id and video_id not in ['', 'unknown', 'error']:
                return video_id
            
            # Method 5: Extract from filename
            filename = json_path.stem
            parts = filename.split('_')
            for part in parts:
                if len(part) == 11 and part.replace('-', '').replace('_', '').isalnum():
                    return part
            
            return None
            
        except Exception as e:
            self.log(f"❌ Error extracting video_id from {json_path.name}: {e}")
            return None
    
    def get_robust_yt_dlp_config(self) -> Dict:
        """Get robust yt-dlp configuration with all OpenAI suggestions"""
        return {
            # Basic extraction settings
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            
            # ROBUST SETTINGS FROM OPENAI SUGGESTIONS
            
            # 1. Android client to avoid desktop checks
            'extractor_args': {
                'youtube': {
                    'player-client': ['android']
                }
            },
            
            # 2. Modern User-Agent
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
            },
            
            # 3. Geo-bypass for region restrictions
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            
            # 4. Enhanced retry logic with exponential backoff
            'retries': 8,
            'fragment_retries': 8,
            'retry_sleep_functions': {
                'http': lambda x: min(600, 2 ** x + random.uniform(0, 1))
            },
            
            # 5. Request rate limiting
            'sleep_interval': 2,
            'max_sleep_interval': 5,
            'sleep_interval_requests': 2,
            'sleep_interval_subtitles': 2,
            
            # 6. Network timeouts
            'socket_timeout': 30,
            
            # 7. Additional robustness
            'nocheckcertificate': False,
            'prefer_insecure': False,
        }
    
    def fetch_enhanced_metadata_robust(self, video_id: str, attempt: int = 1) -> Optional[Dict]:
        """Fetch enhanced metadata with robust error handling and exponential backoff"""
        if not video_id:
            return None
        
        max_attempts = 5
        base_delay = 5
        
        try:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Add jitter to request timing
            if attempt > 1:
                jitter = random.uniform(0.5, 2.0)
                self.log(f"🔄 Retry attempt {attempt} for {video_id}, waiting {jitter:.1f}s...")
                time.sleep(jitter)
            
            # Get robust configuration
            ydl_opts = self.get_robust_yt_dlp_config()
            
            self.log(f"🌐 Fetching metadata for {video_id} (attempt {attempt}/{max_attempts})")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Extract enhanced metadata fields
                enhanced_metadata = {
                    # Engagement metrics
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0), 
                    'channel_follower_count': info.get('channel_follower_count', 0),
                    
                    # Channel info
                    'uploader_id': info.get('uploader_id', ''),
                    'uploader_url': info.get('uploader_url', ''),
                    
                    # Content classification
                    'categories': info.get('categories', []),
                    'availability': info.get('availability', 'public'),
                    'live_status': info.get('live_status', 'not_live'),
                    'age_limit': info.get('age_limit', 0),
                    
                    # Technical data
                    'resolution': info.get('resolution', ''),
                    'fps': info.get('fps', 0),
                    'aspect_ratio': info.get('aspect_ratio', 0.0),
                    'vcodec': info.get('vcodec', ''),
                    'acodec': info.get('acodec', ''),
                    
                    # Caption languages (keys only)
                    'automatic_captions': list(info.get('automatic_captions', {}).keys()),
                    'subtitles': list(info.get('subtitles', {}).keys()),
                    
                    # Timestamps
                    'release_timestamp': info.get('release_timestamp', 0),
                }
                
                self.stats['api_calls'] += 1
                self.log(f"✅ Successfully fetched metadata for {video_id}")
                
                # Rate limiting between requests
                if not self.dry_run:
                    time.sleep(self.request_sleep)
                
                return enhanced_metadata
                
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            
            # Handle specific error types
            if "403" in error_msg or "Forbidden" in error_msg:
                self.stats['rate_limited'] += 1
                self.log(f"🚫 Rate limited for {video_id} (403 Forbidden)")
                
                if attempt < max_attempts:
                    # Exponential backoff with jitter for rate limits
                    delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 5)
                    self.log(f"⏳ Rate limit backoff: waiting {delay:.1f}s before retry...")
                    if not self.dry_run:
                        time.sleep(delay)
                    return self.fetch_enhanced_metadata_robust(video_id, attempt + 1)
                else:
                    self.log(f"❌ Max attempts reached for {video_id} - giving up")
                    return None
            
            elif "429" in error_msg or "Too Many Requests" in error_msg:
                self.stats['rate_limited'] += 1
                self.log(f"🚫 Too many requests for {video_id} (429)")
                
                if attempt < max_attempts:
                    # Longer backoff for 429 errors
                    delay = base_delay * 3 * (2 ** (attempt - 1)) + random.uniform(0, 10)
                    self.log(f"⏳ Heavy rate limit backoff: waiting {delay:.1f}s before retry...")
                    if not self.dry_run:
                        time.sleep(delay)
                    return self.fetch_enhanced_metadata_robust(video_id, attempt + 1)
                else:
                    self.log(f"❌ Max attempts reached for {video_id} - giving up")
                    return None
            
            elif "unavailable" in error_msg.lower() or "private" in error_msg.lower():
                self.log(f"⚠️ Video {video_id} is unavailable or private - skipping")
                return None
                
            elif "timeout" in error_msg.lower() or "network" in error_msg.lower():
                self.stats['network_errors'] += 1
                self.log(f"🌐 Network error for {video_id}: {error_msg}")
                
                if attempt < max_attempts:
                    delay = base_delay + random.uniform(0, 3)
                    self.log(f"⏳ Network retry in {delay:.1f}s...")
                    if not self.dry_run:
                        time.sleep(delay)
                    return self.fetch_enhanced_metadata_robust(video_id, attempt + 1)
                else:
                    return None
            else:
                self.log(f"❌ Unknown error for {video_id}: {error_msg}")
                return None
                
        except Exception as e:
            self.log(f"❌ Unexpected error fetching metadata for {video_id}: {e}")
            return None
    
    def needs_enhancement(self, data: Dict) -> bool:
        """Check if JSON needs enhancement with new metadata fields"""
        if self.force:
            return True
            
        # All remaining files should have universal schema
        # Skip files that don't have the expected structure
        if 'source_metadata' not in data or 'youtube' not in data['source_metadata']:
            self.log(f"⚠️ File missing universal schema - skipping (can reprocess later)")
            return False
            
        youtube_meta = data['source_metadata']['youtube']
        
        # Check for new fields - if any are missing, enhancement is needed
        new_fields = [
            'like_count', 'comment_count', 'channel_follower_count',
            'uploader_id', 'uploader_url', 'categories', 'availability',
            'live_status', 'age_limit', 'resolution', 'fps', 'aspect_ratio',
            'vcodec', 'acodec', 'automatic_captions', 'subtitles', 'release_timestamp'
        ]
        
        for field in new_fields:
            if field not in youtube_meta:
                return True
                
        return False
    
    def update_json_with_enhanced_metadata(self, json_path: Path, enhanced_metadata: Dict) -> bool:
        """Update JSON file with enhanced metadata using atomic write"""
        try:
            # Read current JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Ensure universal schema structure exists
            if 'source_metadata' not in data:
                data['source_metadata'] = {}
            if 'youtube' not in data['source_metadata']:
                data['source_metadata']['youtube'] = {}
            
            # Add/update enhanced fields
            youtube_meta = data['source_metadata']['youtube']
            for key, value in enhanced_metadata.items():
                youtube_meta[key] = value
            
            # Update schema version if exists
            if 'metadata' in data and 'schema_version' in data['metadata']:
                data['metadata']['schema_version'] = "1.1.0"  # Indicate enhancement
            
            if self.dry_run:
                self.log(f"🧪 DRY RUN: Would update {json_path.name} with enhanced metadata")
                return True
            
            # Atomic write: temp file -> rename
            temp_file = str(json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, json_path)
            self.log(f"✅ Enhanced: {json_path.name}")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to update {json_path.name}: {e}")
            return False
    
    def check_connectivity(self) -> bool:
        """Check basic YouTube connectivity before starting"""
        self.log("🔍 Testing YouTube connectivity...")
        
        try:
            # Simple test with a known stable video ID
            test_result = subprocess.run([
                'curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
            ], capture_output=True, text=True, timeout=10)
            
            if test_result.returncode == 0 and test_result.stdout.strip() == '200':
                self.log("✅ YouTube connectivity confirmed")
                return True
            else:
                self.log(f"⚠️ YouTube connectivity issue: HTTP {test_result.stdout.strip()}")
                return False
                
        except Exception as e:
            self.log(f"⚠️ Connectivity test failed: {e}")
            return False
    
    def run_backfill(self, limit: int = None):
        """Run the full backfill process with enhanced robustness"""
        self.log("🚀 Starting YTV2 Metadata Backfill (Enhanced Robust Version)")
        self.log(f"📁 Reports directory: {self.reports_dir}")
        self.log(f"🧪 Dry run: {self.dry_run}")
        self.log(f"🔄 Force update: {self.force}")
        if limit:
            self.log(f"🔢 Limit: {limit} files")
        
        # Check connectivity first
        if not self.dry_run and not self.check_connectivity():
            self.log("❌ Connectivity issues detected. Consider checking your network connection.")
            self.log("💡 You can still run with --dry-run to test the script logic")
            return False
        
        # Load previous state for resume
        self.load_state()
        
        # Get all JSON files, excluding hidden files and backups
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.backfill')
        ]
        
        self.stats['total_files'] = len(json_files)
        self.log(f"📊 Found {len(json_files)} JSON files to process")
        
        processed_in_batch = 0
        consecutive_rate_limits = 0
        
        for json_file in json_files:
            # Check limit
            if limit and self.stats['processed'] >= limit:
                self.log(f"📊 Reached limit of {limit} files")
                break
            
            # Skip if already processed (resume capability)
            if json_file.name in self.processed_files and not self.force:
                self.stats['skipped'] += 1
                continue
            
            self.stats['processed'] += 1
            self.log(f"🔍 Processing {json_file.name} ({self.stats['processed']}/{self.stats['total_files']})")
            
            try:
                # Read and check if enhancement needed
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not self.needs_enhancement(data):
                    self.log(f"⏭️ Skipping {json_file.name} - already enhanced")
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    continue
                
                # Extract video ID
                video_id = self.extract_video_id_from_json(json_file)
                if not video_id:
                    self.log(f"⚠️ No video_id found in {json_file.name} - skipping")
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    continue
                
                # Fetch enhanced metadata with robust handling
                enhanced_metadata = self.fetch_enhanced_metadata_robust(video_id)
                if not enhanced_metadata:
                    self.log(f"⚠️ Could not fetch metadata for {video_id} - skipping")
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    consecutive_rate_limits += 1
                    
                    # If we're getting too many consecutive rate limits, take a longer break
                    if consecutive_rate_limits >= 3:
                        long_pause = 300  # 5 minutes
                        self.log(f"🛑 Too many consecutive rate limits. Taking a {long_pause}s break...")
                        if not self.dry_run:
                            time.sleep(long_pause)
                        consecutive_rate_limits = 0
                    
                    continue
                else:
                    consecutive_rate_limits = 0  # Reset counter on success
                
                # Update JSON file
                if self.update_json_with_enhanced_metadata(json_file, enhanced_metadata):
                    self.stats['updated'] += 1
                else:
                    self.stats['errors'] += 1
                
                # Save state after each successful processing
                self.save_state(json_file.name)
                
                # Enhanced rate limiting with jitter
                processed_in_batch += 1
                if processed_in_batch >= self.batch_size:
                    jitter = random.uniform(0.8, 1.2) * self.sleep_duration
                    self.log(f"⏸️ Batch complete: processed {processed_in_batch} files, sleeping {jitter:.1f}s...")
                    if not self.dry_run:
                        time.sleep(jitter)
                    processed_in_batch = 0
                
            except Exception as e:
                self.log(f"❌ Error processing {json_file.name}: {e}")
                self.stats['errors'] += 1
                self.save_state(json_file.name)
        
        # Final summary
        self.log("\n" + "="*70)
        self.log("ENHANCED ROBUST BACKFILL SUMMARY")
        self.log("="*70)
        self.log(f"📊 Total files found: {self.stats['total_files']}")
        self.log(f"🔄 Files processed: {self.stats['processed']}")
        self.log(f"✅ Files updated: {self.stats['updated']}")
        self.log(f"⏭️ Files skipped: {self.stats['skipped']}")
        self.log(f"❌ Errors: {self.stats['errors']}")
        self.log(f"🌐 API calls made: {self.stats['api_calls']}")
        self.log(f"🚫 Rate limited: {self.stats['rate_limited']}")
        self.log(f"📡 Network errors: {self.stats['network_errors']}")
        
        if self.dry_run:
            self.log("\n🧪 This was a DRY RUN - no files were actually modified")
        
        self.log(f"\n📄 Full log saved to: {self.log_path}")
        
        return self.stats['errors'] == 0


def main():
    parser = argparse.ArgumentParser(description="Enhanced Robust Backfill for YTV2 reports with YouTube metadata")
    parser.add_argument("--reports-dir", default="/Volumes/Docker/YTV2/data/reports",
                        help="Path to reports directory (default: /Volumes/Docker/YTV2/data/reports)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be updated without actually modifying files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Force update even if files already have enhanced metadata")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run using state file")
    
    args = parser.parse_args()
    
    # Validate directory
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    # Initialize backfiller
    backfiller = MetadataBackfiller(
        reports_dir=str(reports_dir),
        dry_run=args.dry_run,
        force=args.force
    )
    
    # Run backfill
    success = backfiller.run_backfill(limit=args.limit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
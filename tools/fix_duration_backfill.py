#!/usr/bin/env python3
"""
YTV2 Duration Fix Backfill Tool
Specifically fixes duration_seconds: 0 in universal schema JSON files
Focuses only on duration data to fix broken files from schema conversion issue
"""

import argparse
import json
import os
import sys
import time
import tempfile
import random
from datetime import datetime
from pathlib import Path

import yt_dlp


class DurationBackfiller:
    def __init__(self, reports_dir, dry_run=False):
        self.reports_dir = Path(reports_dir)
        self.dry_run = dry_run
        
        # State tracking for resume capability
        self.state_file = self.reports_dir / '.duration_backfill_state'
        self.processed_files = set()
        
        # Conservative rate limiting for duration-only requests
        self.batch_size = 15  # Slightly larger batches since we're only getting basic info
        self.sleep_duration = 20  # Conservative sleep between batches
        self.request_sleep = 2   # Sleep between individual requests
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'api_calls': 0,
            'already_fixed': 0,
        }
        
        # Setup logging
        log_file = f"duration_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message, print_also=True):
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
    
    def save_state(self, current_file):
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
    
    def extract_video_id_from_json(self, json_path):
        """Extract video_id from universal schema JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Universal schema location
            if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                video_id = data['source_metadata']['youtube'].get('video_id')
                if video_id and video_id not in ['', 'unknown', 'error']:
                    return video_id
            
            # Fallback to top-level id field 
            if 'id' in data:
                video_id = data['id']
                if video_id and video_id.startswith('yt:'):
                    # Extract from yt:video_id format
                    return video_id.replace('yt:', '')
                elif video_id and len(video_id) == 11:
                    return video_id
            
            return None
            
        except Exception as e:
            self.log(f"❌ Error extracting video_id from {json_path.name}: {e}")
            return None
    
    def get_duration_from_youtube(self, video_id, attempt=1):
        """Fetch ONLY duration from YouTube using minimal yt-dlp config"""
        if not video_id:
            return None
        
        max_attempts = 3
        base_delay = 5
        
        try:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Add jitter to request timing
            if attempt > 1:
                jitter = random.uniform(0.5, 2.0)
                self.log(f"🔄 Retry attempt {attempt} for {video_id}, waiting {jitter:.1f}s...")
                time.sleep(jitter)
            
            # Minimal yt-dlp configuration - only get basic info including duration
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
                
                # Conservative settings for duration-only requests
                'extractor_args': {
                    'youtube': {
                        'player-client': ['android']
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                },
                'retries': 3,
                'socket_timeout': 15,
            }
            
            self.log(f"⏱️ Fetching duration for {video_id} (attempt {attempt}/{max_attempts})")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Extract duration in seconds
                duration = info.get('duration')
                if duration and isinstance(duration, (int, float)) and duration > 0:
                    self.stats['api_calls'] += 1
                    self.log(f"✅ Duration for {video_id}: {int(duration)}s")
                    
                    # Rate limiting between requests
                    if not self.dry_run:
                        time.sleep(self.request_sleep)
                    
                    return int(duration)
                else:
                    self.log(f"⚠️ No valid duration found for {video_id}")
                    return None
                
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            
            # Handle rate limiting
            if "403" in error_msg or "429" in error_msg or "rate" in error_msg.lower():
                self.log(f"🚫 Rate limited for {video_id}")
                
                if attempt < max_attempts:
                    delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 5)
                    self.log(f"⏳ Rate limit backoff: waiting {delay:.1f}s before retry...")
                    if not self.dry_run:
                        time.sleep(delay)
                    return self.get_duration_from_youtube(video_id, attempt + 1)
                else:
                    self.log(f"❌ Max attempts reached for {video_id} - giving up")
                    return None
            
            elif "unavailable" in error_msg.lower() or "private" in error_msg.lower():
                self.log(f"⚠️ Video {video_id} is unavailable or private")
                return None
                
            else:
                self.log(f"❌ Error for {video_id}: {error_msg}")
                return None
                
        except Exception as e:
            self.log(f"❌ Unexpected error fetching duration for {video_id}: {e}")
            return None
    
    def needs_duration_fix(self, data):
        """Check if JSON needs duration fix (has duration_seconds: 0)"""
        
        # Only process universal schema files
        if 'content_source' not in data or data.get('content_source') != 'youtube':
            return False
            
        # Check if duration_seconds is 0 or missing
        duration = data.get('duration_seconds', 0)
        return duration == 0
    
    def update_json_with_duration(self, json_path, duration_seconds):
        """Update JSON file with duration using atomic write"""
        try:
            # Read current JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Update duration_seconds
            data['duration_seconds'] = duration_seconds
            
            # Also update in source_metadata.youtube if it exists
            if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                data['source_metadata']['youtube']['duration'] = duration_seconds
            
            # Update metadata if it exists in legacy location
            if 'metadata' in data and 'duration' in data['metadata']:
                data['metadata']['duration'] = duration_seconds
            
            if self.dry_run:
                self.log(f"🧪 DRY RUN: Would update {json_path.name} with duration: {duration_seconds}s")
                return True
            
            # Atomic write: temp file -> rename
            temp_file = str(json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, json_path)
            self.log(f"✅ Fixed duration: {json_path.name} -> {duration_seconds}s")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to update {json_path.name}: {e}")
            return False
    
    def run_duration_backfill(self, limit=None):
        """Run the duration-specific backfill process"""
        self.log("🚀 Starting YTV2 Duration Fix Backfill")
        self.log(f"📁 Reports directory: {self.reports_dir}")
        self.log(f"🧪 Dry run: {self.dry_run}")
        if limit:
            self.log(f"🔢 Limit: {limit} files")
        
        # Load previous state for resume
        self.load_state()
        
        # Get all JSON files
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.duration')
        ]
        
        self.stats['total_files'] = len(json_files)
        self.log(f"📊 Found {len(json_files)} JSON files to check")
        
        # First pass: identify files that need duration fixes
        files_needing_fix = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if self.needs_duration_fix(data):
                    files_needing_fix.append(json_file)
                else:
                    self.stats['already_fixed'] += 1
                    
            except Exception as e:
                self.log(f"❌ Error checking {json_file.name}: {e}")
                continue
        
        self.log(f"🎯 Found {len(files_needing_fix)} files needing duration fixes")
        self.log(f"✅ {self.stats['already_fixed']} files already have correct duration")
        
        if not files_needing_fix:
            self.log("🎉 No files need duration fixes!")
            return True
        
        # Process files that need duration fixes
        processed_in_batch = 0
        
        for json_file in files_needing_fix:
            # Check limit
            if limit and self.stats['processed'] >= limit:
                self.log(f"📊 Reached limit of {limit} files")
                break
            
            # Skip if already processed (resume capability)
            if json_file.name in self.processed_files:
                self.stats['skipped'] += 1
                continue
            
            self.stats['processed'] += 1
            self.log(f"🔍 Processing {json_file.name} ({self.stats['processed']}/{len(files_needing_fix)})")
            
            try:
                # Extract video ID
                video_id = self.extract_video_id_from_json(json_file)
                if not video_id:
                    self.log(f"⚠️ No video_id found in {json_file.name} - skipping")
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    continue
                
                # Fetch duration from YouTube
                duration = self.get_duration_from_youtube(video_id)
                if not duration:
                    self.log(f"⚠️ Could not fetch duration for {video_id} - skipping")
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    continue
                
                # Update JSON file with duration
                if self.update_json_with_duration(json_file, duration):
                    self.stats['updated'] += 1
                else:
                    self.stats['errors'] += 1
                
                # Save state after each successful processing
                self.save_state(json_file.name)
                
                # Rate limiting with jitter
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
        self.log("\n" + "="*60)
        self.log("DURATION FIX BACKFILL SUMMARY")
        self.log("="*60)
        self.log(f"📊 Total files found: {self.stats['total_files']}")
        self.log(f"✅ Files already correct: {self.stats['already_fixed']}")
        self.log(f"🎯 Files needing fix: {len(files_needing_fix)}")
        self.log(f"🔄 Files processed: {self.stats['processed']}")
        self.log(f"✅ Files updated: {self.stats['updated']}")
        self.log(f"⏭️ Files skipped: {self.stats['skipped']}")
        self.log(f"❌ Errors: {self.stats['errors']}")
        self.log(f"🌐 API calls made: {self.stats['api_calls']}")
        
        if self.dry_run:
            self.log("\n🧪 This was a DRY RUN - no files were actually modified")
        
        self.log(f"\n📄 Full log saved to: {self.log_path}")
        
        return self.stats['errors'] == 0


def main():
    parser = argparse.ArgumentParser(description="Duration Fix Backfill for YTV2 reports")
    parser.add_argument("--reports-dir", default="/Volumes/Docker/YTV2/data/reports",
                        help="Path to reports directory (default: /Volumes/Docker/YTV2/data/reports)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be updated without actually modifying files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate directory
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    # Initialize backfiller
    backfiller = DurationBackfiller(
        reports_dir=str(reports_dir),
        dry_run=args.dry_run
    )
    
    # Run backfill
    success = backfiller.run_duration_backfill(limit=args.limit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
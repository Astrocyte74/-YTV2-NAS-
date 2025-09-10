#!/usr/bin/env python
"""
YTV2 Duration Fix Backfill Tool - Python 2.7 Compatible
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

# Try to import pathlib, fallback to os.path for older Python
try:
    from pathlib import Path
except ImportError:
    # For Python < 3.4, create a minimal Path-like class
    class Path(object):
        def __init__(self, path):
            self._path = str(path)
        
        def __str__(self):
            return self._path
        
        def __div__(self, other):  # For Python 2.7
            return Path(os.path.join(self._path, str(other)))
        
        def __truediv__(self, other):  # For Python 3+
            return self.__div__(other)
        
        @property
        def parent(self):
            return Path(os.path.dirname(self._path))
        
        @property
        def name(self):
            return os.path.basename(self._path)
        
        @property
        def stem(self):
            return os.path.splitext(self.name)[0]
        
        def exists(self):
            return os.path.exists(self._path)
        
        def glob(self, pattern):
            import glob as glob_module
            full_pattern = os.path.join(self._path, pattern)
            return [Path(p) for p in glob_module.glob(full_pattern)]

# Try YouTube-DL first (more widely available), fallback to yt-dlp
try:
    import yt_dlp as ytdl
    YTDL_AVAILABLE = True
except ImportError:
    try:
        import youtube_dl as ytdl
        YTDL_AVAILABLE = True
    except ImportError:
        YTDL_AVAILABLE = False


class DurationBackfiller(object):
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
        log_file = "duration_backfill_{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message, print_also=True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = "[{}] {}".format(timestamp, message)
        
        with open(str(self.log_path), 'a') as f:
            f.write(log_entry + '\n')
        
        if print_also:
            print(log_entry)
    
    def load_state(self):
        """Load previous processing state for resume"""
        if self.state_file.exists():
            try:
                with open(str(self.state_file), 'r') as f:
                    state = json.load(f)
                self.processed_files = set(state.get('processed_files', []))
                self.log("📁 Resumed: {} files already processed".format(len(self.processed_files)))
            except Exception as e:
                self.log("⚠️ Could not load state file: {}".format(e))
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
        os.rename(temp_file, str(self.state_file))
    
    def extract_video_id_from_json(self, json_path):
        """Extract video_id from universal schema JSON file"""
        try:
            with open(str(json_path), 'r') as f:
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
            self.log("❌ Error extracting video_id from {}: {}".format(json_path.name, e))
            return None
    
    def get_duration_from_youtube(self, video_id, attempt=1):
        """Fetch ONLY duration from YouTube using minimal config"""
        if not video_id:
            return None
        
        if not YTDL_AVAILABLE:
            self.log("❌ No YouTube downloader available (yt-dlp or youtube-dl)")
            return None
        
        max_attempts = 3
        base_delay = 5
        
        try:
            youtube_url = "https://www.youtube.com/watch?v={}".format(video_id)
            
            # Add jitter to request timing
            if attempt > 1:
                jitter = random.uniform(0.5, 2.0)
                self.log("🔄 Retry attempt {} for {}, waiting {:.1f}s...".format(attempt, video_id, jitter))
                time.sleep(jitter)
            
            # Minimal yt-dlp/youtube-dl configuration
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
                'socket_timeout': 15,
            }
            
            # Add yt-dlp specific options if available
            if hasattr(ytdl, 'YoutubeDL'):
                if 'yt_dlp' in str(type(ytdl)):
                    ydl_opts.update({
                        'extractor_args': {
                            'youtube': {
                                'player-client': ['android']
                            }
                        },
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        },
                    })
            
            self.log("⏱️ Fetching duration for {} (attempt {}/{})".format(video_id, attempt, max_attempts))
            
            with ytdl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Extract duration in seconds
                duration = info.get('duration')
                if duration and isinstance(duration, (int, float)) and duration > 0:
                    self.stats['api_calls'] += 1
                    self.log("✅ Duration for {}: {}s".format(video_id, int(duration)))
                    
                    # Rate limiting between requests
                    if not self.dry_run:
                        time.sleep(self.request_sleep)
                    
                    return int(duration)
                else:
                    self.log("⚠️ No valid duration found for {}".format(video_id))
                    return None
                
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limiting
            if "403" in error_msg or "429" in error_msg or "rate" in error_msg.lower():
                self.log("🚫 Rate limited for {}".format(video_id))
                
                if attempt < max_attempts:
                    delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 5)
                    self.log("⏳ Rate limit backoff: waiting {:.1f}s before retry...".format(delay))
                    if not self.dry_run:
                        time.sleep(delay)
                    return self.get_duration_from_youtube(video_id, attempt + 1)
                else:
                    self.log("❌ Max attempts reached for {} - giving up".format(video_id))
                    return None
            
            elif "unavailable" in error_msg.lower() or "private" in error_msg.lower():
                self.log("⚠️ Video {} is unavailable or private".format(video_id))
                return None
                
            else:
                self.log("❌ Error for {}: {}".format(video_id, error_msg))
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
            with open(str(json_path), 'r') as f:
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
                self.log("🧪 DRY RUN: Would update {} with duration: {}s".format(json_path.name, duration_seconds))
                return True
            
            # Atomic write: temp file -> rename
            temp_file = str(json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, str(json_path))
            self.log("✅ Fixed duration: {} -> {}s".format(json_path.name, duration_seconds))
            return True
            
        except Exception as e:
            self.log("❌ Failed to update {}: {}".format(json_path.name, e))
            return False
    
    def run_duration_backfill(self, limit=None):
        """Run the duration-specific backfill process"""
        self.log("🚀 Starting YTV2 Duration Fix Backfill")
        self.log("📁 Reports directory: {}".format(self.reports_dir))
        self.log("🧪 Dry run: {}".format(self.dry_run))
        if limit:
            self.log("🔢 Limit: {} files".format(limit))
        
        if not YTDL_AVAILABLE:
            self.log("❌ FATAL: No YouTube downloader available (yt-dlp or youtube-dl required)")
            return False
        
        # Load previous state for resume
        self.load_state()
        
        # Get all JSON files
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.duration')
        ]
        
        self.stats['total_files'] = len(json_files)
        self.log("📊 Found {} JSON files to check".format(len(json_files)))
        
        # First pass: identify files that need duration fixes
        files_needing_fix = []
        for json_file in json_files:
            try:
                with open(str(json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if self.needs_duration_fix(data):
                    files_needing_fix.append(json_file)
                else:
                    self.stats['already_fixed'] += 1
                    
            except Exception as e:
                self.log("❌ Error checking {}: {}".format(json_file.name, e))
                continue
        
        self.log("🎯 Found {} files needing duration fixes".format(len(files_needing_fix)))
        self.log("✅ {} files already have correct duration".format(self.stats['already_fixed']))
        
        if not files_needing_fix:
            self.log("🎉 No files need duration fixes!")
            return True
        
        # Process files that need duration fixes
        processed_in_batch = 0
        
        for json_file in files_needing_fix:
            # Check limit
            if limit and self.stats['processed'] >= limit:
                self.log("📊 Reached limit of {} files".format(limit))
                break
            
            # Skip if already processed (resume capability)
            if json_file.name in self.processed_files:
                self.stats['skipped'] += 1
                continue
            
            self.stats['processed'] += 1
            self.log("🔍 Processing {} ({}/{})".format(json_file.name, self.stats['processed'], len(files_needing_fix)))
            
            try:
                # Extract video ID
                video_id = self.extract_video_id_from_json(json_file)
                if not video_id:
                    self.log("⚠️ No video_id found in {} - skipping".format(json_file.name))
                    self.save_state(json_file.name)
                    self.stats['skipped'] += 1
                    continue
                
                # Fetch duration from YouTube
                duration = self.get_duration_from_youtube(video_id)
                if not duration:
                    self.log("⚠️ Could not fetch duration for {} - skipping".format(video_id))
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
                    self.log("⏸️ Batch complete: processed {} files, sleeping {:.1f}s...".format(processed_in_batch, jitter))
                    if not self.dry_run:
                        time.sleep(jitter)
                    processed_in_batch = 0
                
            except Exception as e:
                self.log("❌ Error processing {}: {}".format(json_file.name, e))
                self.stats['errors'] += 1
                self.save_state(json_file.name)
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("DURATION FIX BACKFILL SUMMARY")
        self.log("="*60)
        self.log("📊 Total files found: {}".format(self.stats['total_files']))
        self.log("✅ Files already correct: {}".format(self.stats['already_fixed']))
        self.log("🎯 Files needing fix: {}".format(len(files_needing_fix)))
        self.log("🔄 Files processed: {}".format(self.stats['processed']))
        self.log("✅ Files updated: {}".format(self.stats['updated']))
        self.log("⏭️ Files skipped: {}".format(self.stats['skipped']))
        self.log("❌ Errors: {}".format(self.stats['errors']))
        self.log("🌐 API calls made: {}".format(self.stats['api_calls']))
        
        if self.dry_run:
            self.log("\n🧪 This was a DRY RUN - no files were actually modified")
        
        self.log("\n📄 Full log saved to: {}".format(self.log_path))
        
        return self.stats['errors'] == 0


def main():
    parser = argparse.ArgumentParser(description="Duration Fix Backfill for YTV2 reports")
    parser.add_argument("--reports-dir", default="/volume1/Docker/YTV2/data/reports",
                        help="Path to reports directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be updated without actually modifying files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate directory
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print("❌ Reports directory not found: {}".format(reports_dir))
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
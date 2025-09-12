#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YTV2 Media Metadata Backfill Tool
Updates existing JSON files with comprehensive media metadata including:
- MP3 duration (using ffprobe for accuracy)
- Reading time estimates 
- Video duration verification
- TTS voice detection
"""

import argparse
import json
import os
import sys
import subprocess
import shutil
import re
from datetime import datetime
from pathlib import Path


class MediaMetadataBackfiller:
    def __init__(self, reports_dir, mp3_dir=None, dry_run=False):
        self.reports_dir = Path(reports_dir)
        self.mp3_dir = Path(mp3_dir) if mp3_dir else self.reports_dir.parent / 'mp3'
        self.dry_run = dry_run
        
        # State tracking for resume capability
        self.state_file = self.reports_dir / '.media_metadata_backfill_state'
        self.processed_files = set()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'already_has_metadata': 0,
            'no_mp3_found': 0,
            'mp3_durations_fixed': 0,
        }
        
        # Setup logging
        log_file = f"media_metadata_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message, print_also=True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        try:
            with open(str(self.log_path), 'a') as f:
                f.write(log_entry + '\n')
        except Exception:
            pass  # Ignore logging errors
        
        if print_also:
            print(log_entry)
    
    def load_state(self):
        """Load previous processing state for resume"""
        if self.state_file.exists():
            try:
                with open(str(self.state_file), 'r') as f:
                    state = json.load(f)
                self.processed_files = set(state.get('processed_files', []))
                self.log(f"RESUMED: {len(self.processed_files)} files already processed")
            except Exception as e:
                self.log(f"WARNING: Could not load state file: {e}")
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
        try:
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            os.rename(temp_file, str(self.state_file))
        except Exception:
            pass  # Ignore state save errors
    
    def get_mp3_duration_seconds(self, mp3_path):
        """Get MP3 duration in seconds using ffprobe"""
        if not mp3_path.exists():
            return None
            
        # Check for ffprobe
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            self.log("ERROR: ffprobe not found in PATH")
            return None
        
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", 
                 "-of", "csv=p=0", str(mp3_path)],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                duration_float = float(result.stdout.strip())
                duration_int = int(round(duration_float))
                return duration_int
            else:
                return None
                
        except Exception as e:
            self.log(f"ERROR: ffprobe failed for {mp3_path.name}: {e}")
            return None
    
    def calculate_reading_time_minutes(self, word_count, wpm=200):
        """Calculate reading time in minutes based on word count"""
        if not word_count or word_count <= 0:
            return None
        return max(1, round(word_count / wpm))
    
    def detect_tts_voice(self, mp3_path):
        """Detect TTS voice from filename pattern"""
        # Pattern: filename_voice.mp3 or filename_voice_chunk001.mp3
        name = mp3_path.stem
        
        # Known voices
        voices = ['fable', 'alloy', 'echo', 'onyx', 'nova', 'shimmer']
        
        for voice in voices:
            if f'_{voice}' in name or name.endswith(voice):
                return voice
        
        return None
    
    def extract_video_id_from_json(self, json_file):
        """Extract YouTube video ID from JSON filename or content"""
        try:
            # Method 1: Extract from JSON content
            with open(str(json_file), 'r') as f:
                data = json.load(f)
            
            # Try source_metadata.youtube.video_id first  
            if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                video_id = data['source_metadata']['youtube'].get('video_id')
                if video_id and video_id not in ['', 'unknown', 'error']:
                    return video_id
            
            # Try metadata.video_id
            if 'metadata' in data and 'video_id' in data['metadata']:
                video_id = data['metadata']['video_id']
                if video_id:
                    return video_id
            
            # Method 2: Extract from filename
            # Pattern: some_title_VideoId.json
            filename = json_file.stem
            
            # YouTube video IDs are 11 characters, alphanumeric + - and _
            # Look for 11-character string at end of filename
            parts = filename.split('_')
            if parts:
                last_part = parts[-1]
                if len(last_part) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', last_part):
                    return last_part
            
            # Alternative: look for pattern in middle of filename
            video_id_pattern = r'([a-zA-Z0-9_-]{11})'
            matches = re.findall(video_id_pattern, filename)
            if matches:
                # Take the last match (likely the video ID)
                return matches[-1]
            
            return None
            
        except Exception as e:
            self.log(f"ERROR: extracting video_id from {json_file.name}: {e}")
            return None
    
    def find_mp3_for_json(self, json_file):
        """Find corresponding MP3 file for a JSON report"""
        # Extract video ID from JSON 
        video_id = self.extract_video_id_from_json(json_file)
        if not video_id:
            self.log(f"  ⚠️ Could not extract video_id from {json_file.name}")
            return None
        
        # Look for MP3 files with this video ID
        # Pattern: audio_{video_id}_{timestamp}.mp3 or {video_id}_{timestamp}.mp3
        patterns = [
            f"audio_{video_id}_*.mp3",
            f"{video_id}_*.mp3",
            f"*_{video_id}_*.mp3"
        ]
        
        for pattern in patterns:
            matches = list(self.mp3_dir.glob(pattern))
            if matches:
                # Skip chunk files
                non_chunks = [m for m in matches if '_chunk' not in m.name]
                if non_chunks:
                    # Take the most recent file if multiple matches
                    return max(non_chunks, key=lambda x: x.stat().st_mtime)
        
        # Fallback: exact stem match
        json_stem = json_file.stem
        exact_mp3 = self.mp3_dir / f"{json_stem}.mp3"
        if exact_mp3.exists():
            return exact_mp3
        
        return None
    
    def extract_word_count(self, data):
        """Extract word count from various locations in JSON"""
        # Try new universal schema location
        if 'word_count' in data:
            return data['word_count']
        
        # Try summary section
        if 'summary' in data and isinstance(data['summary'], dict):
            summary_text = data['summary'].get('summary', '')
            if summary_text:
                # Count words in summary
                words = len(summary_text.split())
                return words
        
        # Try metadata section  
        if 'metadata' in data:
            if 'word_count' in data['metadata']:
                return data['metadata']['word_count']
        
        return None
    
    def update_json_with_media_metadata(self, json_path, mp3_path=None):
        """Update JSON file with media metadata"""
        try:
            # Read current JSON
            with open(str(json_path), 'r') as f:
                data = json.load(f)
            
            # Initialize or get existing media_metadata
            if 'media_metadata' not in data:
                data['media_metadata'] = {}
                self.log(f"Creating media_metadata for {json_path.name}")
            
            media_meta = data['media_metadata']
            updates_made = False
            
            # 1. Update MP3 metadata if MP3 exists
            if mp3_path and mp3_path.exists():
                mp3_duration = self.get_mp3_duration_seconds(mp3_path)
                if mp3_duration:
                    if media_meta.get('mp3_duration_seconds') != mp3_duration:
                        media_meta['mp3_duration_seconds'] = mp3_duration
                        updates_made = True
                        self.stats['mp3_durations_fixed'] += 1
                        self.log(f"  ✅ MP3 duration: {mp3_duration}s")
                
                # Detect voice if not set
                if 'mp3_voice' not in media_meta or not media_meta['mp3_voice']:
                    voice = self.detect_tts_voice(mp3_path)
                    if voice:
                        media_meta['mp3_voice'] = voice
                        updates_made = True
                        self.log(f"  ✅ TTS voice: {voice}")
                
                # Mark as having audio
                if not media_meta.get('has_audio'):
                    media_meta['has_audio'] = True
                    updates_made = True
                
                # Add creation timestamp if missing
                if 'mp3_created_at' not in media_meta:
                    # Use file modification time as approximation
                    mp3_mtime = os.path.getmtime(str(mp3_path))
                    media_meta['mp3_created_at'] = datetime.fromtimestamp(mp3_mtime).isoformat()
                    updates_made = True
            
            # 2. Calculate reading time from word count
            word_count = self.extract_word_count(data)
            if word_count and word_count > 0:
                reading_time = self.calculate_reading_time_minutes(word_count)
                if reading_time and media_meta.get('estimated_reading_minutes') != reading_time:
                    media_meta['estimated_reading_minutes'] = reading_time
                    updates_made = True
                    self.log(f"  ✅ Reading time: {reading_time} min ({word_count} words)")
            
            # 3. Verify/update video duration if YouTube content
            if data.get('content_source') == 'youtube':
                # Check if duration_seconds is in root
                video_duration = data.get('duration_seconds')
                if video_duration and video_duration > 0:
                    if media_meta.get('video_duration_seconds') != video_duration:
                        media_meta['video_duration_seconds'] = video_duration
                        updates_made = True
                        self.log(f"  ✅ Video duration: {video_duration}s")
                
                # Also check metadata.duration as fallback
                elif 'metadata' in data and 'duration' in data['metadata']:
                    video_duration = data['metadata']['duration']
                    if video_duration and video_duration > 0:
                        media_meta['video_duration_seconds'] = video_duration
                        updates_made = True
                        self.log(f"  ✅ Video duration: {video_duration}s (from metadata)")
            
            # 4. Add processing timestamp
            if updates_made:
                media_meta['last_updated'] = datetime.now().isoformat()
            
            if not updates_made:
                self.log(f"  ℹ️ No updates needed for {json_path.name}")
                return False
            
            if self.dry_run:
                self.log(f"DRY RUN: Would update {json_path.name}")
                return True
            
            # Atomic write: temp file -> rename
            temp_file = str(json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, str(json_path))
            self.log(f"UPDATED: {json_path.name}")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Failed to update {json_path.name}: {e}")
            return False
    
    def run_backfill(self, limit=None):
        """Run the media metadata backfill process"""
        self.log("=" * 60)
        self.log("STARTING: YTV2 Media Metadata Backfill")
        self.log(f"REPORTS DIR: {self.reports_dir}")
        self.log(f"MP3 DIR: {self.mp3_dir}")
        self.log(f"DRY RUN: {self.dry_run}")
        
        # Check for ffprobe
        if not shutil.which("ffprobe"):
            self.log("WARNING: ffprobe not found - MP3 duration extraction will fail")
            self.log("Install ffmpeg to enable MP3 duration extraction")
        
        # Load previous state for resume
        self.load_state()
        
        # Get all JSON files
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.media')
        ]
        
        self.stats['total_files'] = len(json_files)
        self.log(f"FOUND: {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for idx, json_file in enumerate(json_files, 1):
            # Check limit
            if limit and self.stats['processed'] >= limit:
                self.log(f"LIMIT REACHED: {limit} files")
                break
            
            # Skip if already processed (resume capability)
            if json_file.name in self.processed_files:
                self.stats['skipped'] += 1
                continue
            
            self.stats['processed'] += 1
            self.log(f"\nPROCESSING: {json_file.name} ({idx}/{len(json_files)})")
            
            try:
                # Check if already has complete metadata
                with open(str(json_file), 'r') as f:
                    data = json.load(f)
                
                has_complete_metadata = (
                    'media_metadata' in data and
                    data['media_metadata'].get('mp3_duration_seconds') and
                    data['media_metadata'].get('estimated_reading_minutes')
                )
                
                if has_complete_metadata and not self.dry_run:
                    self.log(f"  ℹ️ Already has complete metadata")
                    self.stats['already_has_metadata'] += 1
                    self.save_state(json_file.name)
                    continue
                
                # Find corresponding MP3
                mp3_file = self.find_mp3_for_json(json_file)
                if mp3_file:
                    self.log(f"  📎 Found MP3: {mp3_file.name}")
                else:
                    self.log(f"  ⚠️ No MP3 found")
                    self.stats['no_mp3_found'] += 1
                
                # Update metadata
                if self.update_json_with_media_metadata(json_file, mp3_file):
                    self.stats['updated'] += 1
                else:
                    self.stats['errors'] += 1
                
                # Save state after each file
                self.save_state(json_file.name)
                
            except Exception as e:
                self.log(f"ERROR: processing {json_file.name}: {e}")
                self.stats['errors'] += 1
                self.save_state(json_file.name)
        
        # Final summary
        self.log("")
        self.log("=" * 60)
        self.log("MEDIA METADATA BACKFILL SUMMARY")
        self.log("=" * 60)
        self.log(f"Total files found: {self.stats['total_files']}")
        self.log(f"Files processed: {self.stats['processed']}")
        self.log(f"Files updated: {self.stats['updated']}")
        self.log(f"Files skipped: {self.stats['skipped']}")
        self.log(f"Already had metadata: {self.stats['already_has_metadata']}")
        self.log(f"No MP3 found: {self.stats['no_mp3_found']}")
        self.log(f"MP3 durations fixed: {self.stats['mp3_durations_fixed']}")
        self.log(f"Errors: {self.stats['errors']}")
        
        if self.dry_run:
            self.log("")
            self.log("DRY RUN: This was a test - no files were actually modified")
        
        self.log("")
        self.log(f"Full log saved to: {self.log_path}")
        
        return self.stats['errors'] == 0


def main():
    parser = argparse.ArgumentParser(
        description="Backfill media metadata for YTV2 reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files
  python backfill_media_metadata.py
  
  # Dry run to see what would be updated
  python backfill_media_metadata.py --dry-run
  
  # Process only first 10 files
  python backfill_media_metadata.py --limit 10
  
  # Use custom directories
  python backfill_media_metadata.py --reports-dir /path/to/reports --mp3-dir /path/to/mp3

The script will:
- Extract accurate MP3 durations using ffprobe
- Calculate reading time estimates from word counts
- Detect TTS voice from MP3 filenames
- Update video durations from YouTube metadata
- Support resume capability if interrupted
"""
    )
    
    parser.add_argument("--reports-dir", 
                        default="/Users/markdarby/projects/YTV_temp_NAS_files/data/reports",
                        help="Path to reports directory containing JSON files")
    parser.add_argument("--mp3-dir",
                        help="Path to MP3 directory (default: reports_dir/../mp3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be updated without modifying files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate directories
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"ERROR: Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    mp3_dir = args.mp3_dir
    if not mp3_dir:
        # Default to sibling mp3 directory
        mp3_dir = reports_dir.parent / 'mp3'
    else:
        mp3_dir = Path(mp3_dir)
    
    if not mp3_dir.exists():
        print(f"WARNING: MP3 directory not found: {mp3_dir}")
        print("Will continue but won't be able to extract MP3 durations")
    
    # Initialize backfiller
    backfiller = MediaMetadataBackfiller(
        reports_dir=str(reports_dir),
        mp3_dir=str(mp3_dir),
        dry_run=args.dry_run
    )
    
    # Run backfill
    success = backfiller.run_backfill(limit=args.limit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
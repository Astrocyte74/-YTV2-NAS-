#!/usr/bin/env python3
"""
YTV2 Video ID Fix Tool
Extracts video IDs from filenames and updates empty video_id fields in JSON files
Prepares files for metadata backfill by ensuring valid video IDs
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class VideoIdFixer:
    def __init__(self, reports_dir: str, dry_run: bool = False):
        self.reports_dir = Path(reports_dir)
        self.dry_run = dry_run
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'fixed': 0,
            'already_valid': 0,
            'no_video_id_found': 0,
            'errors': 0,
        }
        
        # Setup logging
        log_file = f"fix_video_ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message: str, print_also: bool = True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
        
        if print_also:
            print(log_entry)
    
    def extract_video_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract YouTube video ID from filename"""
        # Remove extension
        stem = filename.replace('.json', '')
        
        # Pattern 1: Look for 11-char sequences in underscore-separated parts
        parts = stem.split('_')
        for part in parts:
            if len(part) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', part):
                return part
        
        # Pattern 2: Look for 11-char video ID anywhere in filename
        match = re.search(r'[a-zA-Z0-9_-]{11}', stem)
        if match:
            return match.group()
        
        return None
    
    def get_current_video_id(self, data: dict) -> Optional[str]:
        """Get current video_id from JSON data"""
        # Check universal schema location
        if ('source_metadata' in data and 
            'youtube' in data['source_metadata'] and
            'video_id' in data['source_metadata']['youtube']):
            vid_id = data['source_metadata']['youtube']['video_id']
            if vid_id and vid_id.strip() and not vid_id.startswith('yt:unknown'):
                return vid_id
        
        # Check other possible locations
        if 'video_id' in data:
            vid_id = data['video_id']
            if vid_id and vid_id.strip() and not vid_id.startswith('yt:unknown'):
                return vid_id
        
        return None
    
    def fix_json_video_id(self, json_path: Path) -> bool:
        """Fix video_id in JSON file using filename extraction"""
        try:
            # Read current JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if video_id already exists and is valid
            current_video_id = self.get_current_video_id(data)
            if current_video_id:
                self.log(f"✅ {json_path.name} already has valid video_id: {current_video_id}")
                self.stats['already_valid'] += 1
                return True
            
            # Extract video_id from filename
            extracted_video_id = self.extract_video_id_from_filename(json_path.name)
            if not extracted_video_id:
                self.log(f"⚠️ {json_path.name} - Could not extract video_id from filename")
                self.stats['no_video_id_found'] += 1
                return True  # Not an error, just can't fix
            
            # Ensure universal schema structure exists
            if 'source_metadata' not in data:
                data['source_metadata'] = {}
            if 'youtube' not in data['source_metadata']:
                data['source_metadata']['youtube'] = {}
            
            # Update video_id
            data['source_metadata']['youtube']['video_id'] = extracted_video_id
            
            if self.dry_run:
                self.log(f"🧪 DRY RUN: Would fix {json_path.name} with video_id: {extracted_video_id}")
                self.stats['fixed'] += 1
                return True
            
            # Atomic write: temp file -> rename
            temp_file = str(json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, json_path)
            self.log(f"✅ Fixed: {json_path.name} -> video_id: {extracted_video_id}")
            self.stats['fixed'] += 1
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to fix {json_path.name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def run_fix(self, limit: int = None):
        """Run the video ID fix process"""
        self.log("🚀 Starting YTV2 Video ID Fix")
        self.log(f"📁 Reports directory: {self.reports_dir}")
        self.log(f"🧪 Dry run: {self.dry_run}")
        if limit:
            self.log(f"🔢 Limit: {limit} files")
        
        # Get all JSON files, excluding hidden files and backups
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [
            f for f in all_json_files 
            if not f.name.startswith('._') and not f.name.startswith('.fix')
        ]
        
        self.stats['total_files'] = len(json_files)
        self.log(f"📊 Found {len(json_files)} JSON files to process")
        
        for i, json_file in enumerate(json_files, 1):
            # Check limit
            if limit and i > limit:
                self.log(f"📊 Reached limit of {limit} files")
                break
            
            self.log(f"🔍 Processing {json_file.name} ({i}/{len(json_files)})")
            self.fix_json_video_id(json_file)
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("VIDEO ID FIX SUMMARY")
        self.log("="*60)
        self.log(f"📊 Total files processed: {self.stats['total_files']}")
        self.log(f"✅ Files fixed: {self.stats['fixed']}")
        self.log(f"✅ Already valid: {self.stats['already_valid']}")
        self.log(f"⚠️ No video_id found: {self.stats['no_video_id_found']}")
        self.log(f"❌ Errors: {self.stats['errors']}")
        
        if self.dry_run:
            self.log("\n🧪 This was a DRY RUN - no files were actually modified")
        
        self.log(f"\n📄 Full log saved to: {self.log_path}")
        
        return self.stats['errors'] == 0


def main():
    parser = argparse.ArgumentParser(description="Fix empty video_id fields in YTV2 JSON reports using filename extraction")
    parser.add_argument("--reports-dir", default="/Volumes/Docker/YTV2/data/reports",
                        help="Path to reports directory (default: /Volumes/Docker/YTV2/data/reports)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be fixed without actually modifying files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate directory
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    # Initialize fixer
    fixer = VideoIdFixer(
        reports_dir=str(reports_dir),
        dry_run=args.dry_run
    )
    
    # Run fix
    success = fixer.run_fix(limit=args.limit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
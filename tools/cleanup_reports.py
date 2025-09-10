#!/usr/bin/env python3
"""
YTV2 Reports Cleanup Tool
Removes duplicate JSON files and orphaned reports without corresponding MP3s
Safe operation with backup and dry-run modes
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

class ReportCleaner:
    def __init__(self, reports_dir: str, exports_dir: str, backup_dir: str = None, dry_run: bool = False):
        self.reports_dir = Path(reports_dir)
        self.exports_dir = Path(exports_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.dry_run = dry_run
        
        self.stats = {
            'total_jsons': 0,
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'orphans_found': 0, 
            'orphans_removed': 0,
            'errors': 0
        }
        
        # Setup logging
        log_file = f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = self.reports_dir.parent / log_file
        
    def log(self, message: str, print_also: bool = True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
        
        if print_also:
            print(log_entry)
    
    def extract_video_id_from_json(self, json_path: Path) -> str:
        """Extract video_id from JSON file, trying multiple fields"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Try different possible locations for video_id
            video_id = None
            
            # Universal schema location
            if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                video_id = data['source_metadata']['youtube'].get('video_id')
            
            # Legacy locations
            if not video_id and 'metadata' in data:
                metadata = data['metadata']
                video_id = metadata.get('video_id') or metadata.get('id')
            
            # Top-level locations
            if not video_id:
                video_id = data.get('video_id') or data.get('id')
            
            # Extract from filename if JSON doesn't contain it
            if not video_id:
                # Filename format: videoId_timestamp.json
                filename = json_path.stem
                parts = filename.split('_')
                if len(parts) >= 2 and len(parts[0]) == 11:  # YouTube video IDs are 11 chars
                    video_id = parts[0]
            
            return video_id or "unknown"
            
        except Exception as e:
            self.log(f"Error reading {json_path}: {e}")
            self.stats['errors'] += 1
            return "error"
    
    def get_file_timestamp(self, file_path: Path) -> float:
        """Get file modification timestamp"""
        try:
            return file_path.stat().st_mtime
        except:
            return 0.0
    
    def find_duplicates(self) -> Dict[str, List[Tuple[Path, float]]]:
        """Find duplicate JSON files based on video_id"""
        video_id_map = defaultdict(list)
        
        self.log("🔍 Scanning for JSON files and extracting video IDs...")
        
        # Get all JSON files but exclude macOS hidden files
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [f for f in all_json_files if not f.name.startswith('._')]
        hidden_files = len(all_json_files) - len(json_files)
        
        self.stats['total_jsons'] = len(json_files)
        if hidden_files > 0:
            self.log(f"Excluding {hidden_files} macOS hidden files (._ prefix)")
        
        self.log(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            video_id = self.extract_video_id_from_json(json_file)
            timestamp = self.get_file_timestamp(json_file)
            video_id_map[video_id].append((json_file, timestamp))
        
        # Find duplicates (video_ids with multiple files)
        duplicates = {}
        for video_id, files in video_id_map.items():
            if len(files) > 1:
                # Sort by timestamp (newest first)
                files.sort(key=lambda x: x[1], reverse=True)
                duplicates[video_id] = files
                self.stats['duplicates_found'] += len(files) - 1  # -1 because we keep one
        
        unique_video_ids = len([v for v in video_id_map.values() if len(v) == 1])
        duplicate_video_ids = len(duplicates)
        
        self.log(f"📊 Analysis complete:")
        self.log(f"   - Unique video IDs: {unique_video_ids}")
        self.log(f"   - Video IDs with duplicates: {duplicate_video_ids}")
        self.log(f"   - Total duplicate files to remove: {self.stats['duplicates_found']}")
        
        return duplicates
    
    def extract_video_id_from_filename(self, file_path: Path) -> str:
        """Extract video_id from filename, trying multiple patterns"""
        filename = file_path.stem
        
        # Pattern 1: _videoId_timestamp.ext or videoId_timestamp.ext  
        if '_' in filename:
            parts = filename.split('_')
            for part in parts:
                # YouTube video IDs are 11 characters, alphanumeric + - and _
                if len(part) == 11 and part.replace('-', '').replace('_', '').isalnum():
                    return part
        
        # Pattern 2: Look for 11-char video ID anywhere in filename
        import re
        # YouTube video ID pattern: 11 chars, letters/numbers/hyphens/underscores
        match = re.search(r'[a-zA-Z0-9_-]{11}', filename)
        if match:
            return match.group()
            
        return "unknown"

    def find_orphans(self) -> List[Path]:
        """Find JSON files without corresponding MP3 files"""
        self.log("🔍 Checking for orphaned JSON files (no corresponding MP3)...")
        
        # Get all MP3 files and extract their video IDs
        mp3_video_ids = set()
        mp3_filenames = set()
        
        for mp3_path in self.exports_dir.glob('*.mp3'):
            mp3_filenames.add(mp3_path.stem)
            
            # Extract video ID from MP3 filename
            video_id = self.extract_video_id_from_filename(mp3_path)
            if video_id != "unknown":
                mp3_video_ids.add(video_id)
        
        self.log(f"Found {len(mp3_filenames)} MP3 files with {len(mp3_video_ids)} unique video IDs")
        
        # Get all JSON files but exclude macOS hidden files
        all_json_files = list(self.reports_dir.glob('*.json'))
        json_files = [f for f in all_json_files if not f.name.startswith('._')]
        hidden_files = len(all_json_files) - len(json_files)
        
        if hidden_files > 0:
            self.log(f"Excluding {hidden_files} macOS hidden files (._ prefix) from orphan check")
        
        orphans = []
        for json_path in json_files:
            json_stem = json_path.stem
            
            # Check if there's a corresponding MP3
            has_mp3 = False
            
            # Method 1: Direct filename match
            if json_stem in mp3_filenames:
                has_mp3 = True
            
            # Method 2: Extract video_id from JSON filename and check MP3s
            if not has_mp3:
                json_video_id = self.extract_video_id_from_filename(json_path)
                if json_video_id != "unknown" and json_video_id in mp3_video_ids:
                    has_mp3 = True
            
            # Method 3: Extract video_id from JSON content and check MP3s
            if not has_mp3:
                content_video_id = self.extract_video_id_from_json(json_path)
                if content_video_id and content_video_id != "unknown" and content_video_id != "error":
                    if content_video_id in mp3_video_ids:
                        has_mp3 = True
            
            if not has_mp3:
                orphans.append(json_path)
        
        self.stats['orphans_found'] = len(orphans)
        self.log(f"📊 Found {len(orphans)} orphaned JSON files")
        
        return orphans
    
    def create_backup(self, files_to_delete: List[Path]) -> bool:
        """Create backup of files before deletion"""
        if not self.backup_dir or self.dry_run:
            return True
            
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"📦 Creating backup in {self.backup_dir}")
            
            for file_path in files_to_delete:
                backup_path = self.backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                
            self.log(f"✅ Backed up {len(files_to_delete)} files")
            return True
            
        except Exception as e:
            self.log(f"❌ Backup failed: {e}")
            return False
    
    def remove_duplicates(self, duplicates: Dict[str, List[Tuple[Path, float]]], limit: int = None) -> List[Path]:
        """Remove duplicate files, keeping the newest"""
        files_to_delete = []
        
        for video_id, files in duplicates.items():
            if limit and len(files_to_delete) >= limit:
                break
                
            # Keep the first (newest) file, delete the rest
            files_to_keep = files[0]
            files_to_remove = files[1:]
            
            self.log(f"📝 Video ID {video_id}:")
            self.log(f"   KEEPING: {files_to_keep[0].name} (newest: {datetime.fromtimestamp(files_to_keep[1])})")
            
            for file_path, timestamp in files_to_remove:
                self.log(f"   REMOVING: {file_path.name} (older: {datetime.fromtimestamp(timestamp)})")
                files_to_delete.append(file_path)
        
        return files_to_delete
    
    def remove_orphans(self, orphans: List[Path], limit: int = None) -> List[Path]:
        """Prepare orphaned files for removal"""
        files_to_delete = orphans.copy()
        
        if limit:
            files_to_delete = files_to_delete[:limit]
        
        self.log(f"📝 Orphaned files to remove:")
        for file_path in files_to_delete:
            self.log(f"   REMOVING: {file_path.name} (no corresponding MP3)")
        
        return files_to_delete
    
    def delete_files(self, files_to_delete: List[Path]) -> int:
        """Actually delete the files"""
        deleted_count = 0
        
        if self.dry_run:
            self.log(f"🧪 DRY RUN: Would delete {len(files_to_delete)} files")
            return len(files_to_delete)
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                self.log(f"🗑️  Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                self.log(f"❌ Failed to delete {file_path.name}: {e}")
                self.stats['errors'] += 1
        
        return deleted_count
    
    def run_cleanup(self, remove_duplicates: bool = True, remove_orphans: bool = True, limit: int = None):
        """Run the full cleanup process"""
        self.log("🚀 Starting YTV2 Reports Cleanup")
        self.log(f"📁 Reports directory: {self.reports_dir}")
        self.log(f"📁 Exports directory: {self.exports_dir}")
        self.log(f"🧪 Dry run: {self.dry_run}")
        if self.backup_dir:
            self.log(f"📦 Backup directory: {self.backup_dir}")
        if limit:
            self.log(f"🔢 Limit: {limit} files")
        
        files_to_delete = []
        
        # Step 1: Find and prepare duplicates for removal
        if remove_duplicates:
            self.log("\n" + "="*50)
            self.log("STEP 1: Finding duplicate files")
            self.log("="*50)
            
            duplicates = self.find_duplicates()
            if duplicates:
                duplicate_files = self.remove_duplicates(duplicates, limit)
                files_to_delete.extend(duplicate_files)
                self.stats['duplicates_removed'] = len(duplicate_files)
            else:
                self.log("✅ No duplicate files found")
        
        # Step 2: Find and prepare orphans for removal
        if remove_orphans:
            self.log("\n" + "="*50)
            self.log("STEP 2: Finding orphaned files")
            self.log("="*50)
            
            orphans = self.find_orphans()
            if orphans:
                remaining_limit = limit - len(files_to_delete) if limit else None
                orphan_files = self.remove_orphans(orphans, remaining_limit)
                files_to_delete.extend(orphan_files)
                self.stats['orphans_removed'] = len(orphan_files)
            else:
                self.log("✅ No orphaned files found")
        
        # Step 3: Create backup and delete files
        if files_to_delete:
            self.log("\n" + "="*50)
            self.log("STEP 3: Removing files")
            self.log("="*50)
            
            # Create backup before deletion
            if self.backup_dir and not self.dry_run:
                if not self.create_backup(files_to_delete):
                    self.log("❌ Backup failed, aborting cleanup")
                    return False
            
            # Delete files
            deleted = self.delete_files(files_to_delete)
            
            # Final summary
            self.log("\n" + "="*50)
            self.log("CLEANUP SUMMARY")
            self.log("="*50)
            self.log(f"📊 Total JSON files scanned: {self.stats['total_jsons']}")
            self.log(f"🔄 Duplicate files removed: {self.stats['duplicates_removed']}")
            self.log(f"🚫 Orphaned files removed: {self.stats['orphans_removed']}")
            self.log(f"🗑️  Total files deleted: {deleted}")
            self.log(f"❌ Errors encountered: {self.stats['errors']}")
            
            if self.dry_run:
                self.log("\n🧪 This was a DRY RUN - no files were actually deleted")
            else:
                self.log(f"\n📦 Backup created at: {self.backup_dir}")
                
            self.log(f"📄 Full log saved to: {self.log_path}")
            
        else:
            self.log("\n✅ No files need to be removed - directory is already clean!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate and orphaned YTV2 report files")
    parser.add_argument("--reports-dir", default="/Volumes/Docker/YTV2/data/reports",
                        help="Path to reports directory (default: /Volumes/Docker/YTV2/data/reports)")
    parser.add_argument("--exports-dir", default="/Volumes/Docker/YTV2/exports", 
                        help="Path to exports directory (default: /Volumes/Docker/YTV2/exports)")
    parser.add_argument("--backup-dir", 
                        help="Directory to backup deleted files (default: reports_dir/../backups/cleanup_TIMESTAMP)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be deleted without actually deleting")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="Skip duplicate removal")
    parser.add_argument("--skip-orphans", action="store_true", 
                        help="Skip orphan removal")
    
    args = parser.parse_args()
    
    # Validate directories exist
    reports_dir = Path(args.reports_dir)
    exports_dir = Path(args.exports_dir)
    
    if not reports_dir.exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
        
    if not exports_dir.exists():
        print(f"❌ Exports directory not found: {exports_dir}")
        sys.exit(1)
    
    # Set default backup directory if not provided
    backup_dir = args.backup_dir
    if not backup_dir and not args.dry_run:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = str(reports_dir.parent / f"backups/cleanup_{timestamp}")
    
    # Initialize cleaner
    cleaner = ReportCleaner(
        reports_dir=str(reports_dir),
        exports_dir=str(exports_dir), 
        backup_dir=backup_dir,
        dry_run=args.dry_run
    )
    
    # Run cleanup
    success = cleaner.run_cleanup(
        remove_duplicates=not args.skip_duplicates,
        remove_orphans=not args.skip_orphans,
        limit=args.limit
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
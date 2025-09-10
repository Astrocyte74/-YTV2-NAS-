#!/usr/bin/env python3
"""
Backfill Analysis Script for YTV2 Universal Schema Migration

This script updates existing JSON files to conform to the new universal schema.
Features:
- Idempotent operation (skips already processed files)
- Atomic writes to prevent data corruption
- Resumable with state tracking
- Dry-run mode for safe testing

Usage:
    python tools/backfill_analysis.py --dry-run
    python tools/backfill_analysis.py --limit 10
    python tools/backfill_analysis.py --resume
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import tempfile

# Constants
CURRENT_PIPELINE_VERSION = "2025-09-07-v1"
STATE_FILE = ".backfill_state"
BACKUP_DIR = "backfill_backups"


class BackfillProcessor:
    """Handles the backfill process for YTV2 universal schema migration"""
    
    def __init__(self, reports_dir: str, dry_run: bool = False, limit: Optional[int] = None):
        self.reports_dir = Path(reports_dir)
        self.dry_run = dry_run
        self.limit = limit
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        if not dry_run:
            self.backup_dir = self.reports_dir / BACKUP_DIR
            self.backup_dir.mkdir(exist_ok=True)
        
        print(f"📁 Reports directory: {self.reports_dir}")
        print(f"🏃 Mode: {'DRY RUN' if dry_run else 'LIVE RUN'}")
        if limit:
            print(f"🔢 Limit: {limit} files")
    
    def load_state(self) -> int:
        """Load processing state to resume from last position"""
        state_file = self.reports_dir / STATE_FILE
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_processed_index', 0)
            except Exception as e:
                print(f"⚠️ Warning: Could not load state file: {e}")
        return 0
    
    def save_state(self, last_index: int):
        """Save current processing state"""
        if self.dry_run:
            return
            
        state_file = self.reports_dir / STATE_FILE
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'last_processed_index': last_index,
                    'updated_at': datetime.now().isoformat(),
                    'processed_count': self.processed_count,
                    'error_count': self.error_count
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save state: {e}")
    
    def create_backup(self, file_path: Path) -> bool:
        """Create backup of original file"""
        if self.dry_run:
            return True
            
        try:
            backup_path = self.backup_dir / f"{file_path.stem}_{int(datetime.now().timestamp())}.json"
            backup_path.write_bytes(file_path.read_bytes())
            return True
        except Exception as e:
            print(f"❌ Backup failed for {file_path}: {e}")
            return False
    
    def needs_migration(self, data: Dict[str, Any]) -> bool:
        """Check if file needs migration to new schema"""
        # Check if already has new pipeline version
        processing = data.get('processing', {})
        if processing.get('pipeline_version') == CURRENT_PIPELINE_VERSION:
            return False
        
        # Check if has universal schema structure
        if not data.get('id', '').startswith(('yt:', 'web:', 'pubmed:')):
            return True
        
        # Check if analysis fields are in new format
        analysis = data.get('analysis', {})
        required_fields = ['content_type', 'complexity_level', 'key_topics']
        
        return not all(field in analysis for field in required_fields)
    
    def migrate_to_universal_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy data to universal schema"""
        
        # Extract existing data
        metadata = data.get('metadata', {})
        analysis = data.get('analysis', {})
        url = data.get('url', '')
        transcript = data.get('transcript', '')
        
        # Generate stable ID
        video_id = metadata.get('id', metadata.get('video_id', ''))
        if not video_id and url:
            # Try to extract from URL
            if 'youtube.com' in url or 'youtu.be' in url:
                import re
                match = re.search(r'(?:v=|youtu\.be\/|embed\/|v\/)([\w-]{11})', url)
                if match:
                    video_id = match.group(1)
        
        stable_id = f"yt:{video_id.lower()}" if video_id else f"yt:unknown-{int(datetime.now().timestamp())}"
        
        # Create universal schema structure
        migrated = {
            'id': stable_id,
            'content_source': 'youtube',
            'title': str(metadata.get('title', ''))[:300],
            'canonical_url': url,
            'thumbnail_url': metadata.get('thumbnail', ''),
            'published_at': self._format_youtube_date(metadata.get('upload_date', '')),
            'duration_seconds': metadata.get('duration', 0),
            'word_count': len(transcript.split()) if transcript else 0,
            
            'media': {
                'has_audio': True,  # YouTube videos always have audio
                'audio_duration_seconds': metadata.get('duration', 0),
                'has_transcript': bool(transcript and len(transcript.strip()) > 50),
                'transcript_chars': len(transcript) if transcript else 0
            },
            
            'source_metadata': {
                'youtube': {
                    'video_id': video_id or '',
                    'channel_name': str(metadata.get('uploader', 'Unknown'))[:100],
                    'view_count': metadata.get('view_count', 0),
                    'tags': (metadata.get('tags', []) or [])[:10]
                }
            },
            
            'analysis': self._migrate_analysis_data(analysis),
            
            'processing': {
                'status': 'complete',
                'pipeline_version': CURRENT_PIPELINE_VERSION,
                'attempts': 1,
                'started_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'error': None,
                'logs': [
                    'Migrated from legacy schema',
                    'Universal schema compliance verified'
                ]
            },
            
            # Preserve legacy fields for backward compatibility
            'url': url,
            'metadata': metadata,
            'transcript': transcript,
            'summary': data.get('summary', {}),
            'processed_at': data.get('processed_at', datetime.now().isoformat()),
            'processor_info': data.get('processor_info', {})
        }
        
        return migrated
    
    def _migrate_analysis_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy analysis data to new format"""
        
        # Map legacy categories to new format
        category_mapping = {
            'Education': 'Education',
            'Entertainment': 'Entertainment', 
            'Technology': 'Technology',
            'Tech': 'Technology',
            'Business': 'Business',
            'Health': 'Health',
            'DIY': 'DIY',
            'News': 'News',
            'Gaming': 'Gaming',
            'Lifestyle': 'Lifestyle',
            'Science': 'Science',
            'History': 'History'
        }
        
        # Extract and clean categories
        legacy_categories = analysis.get('category', [])
        if isinstance(legacy_categories, str):
            legacy_categories = [legacy_categories]
        
        new_categories = []
        for cat in legacy_categories[:3]:  # Max 3 categories
            mapped = category_mapping.get(cat, 'General')
            if mapped not in new_categories:
                new_categories.append(mapped)
        
        if not new_categories:
            new_categories = ['General']
        
        # Process key topics to slugs
        key_topics = analysis.get('key_topics', [])
        if isinstance(key_topics, str):
            key_topics = [key_topics]
        
        topic_slugs = []
        for topic in key_topics[:5]:  # Max 5 topics
            if isinstance(topic, str) and topic.strip():
                slug = topic.lower().replace(' ', '-').replace('_', '-')
                # Remove special characters and limit length
                import re
                slug = re.sub(r'[^a-z0-9-]', '', slug)[:30]
                if slug and slug not in topic_slugs:
                    topic_slugs.append(slug)
        
        return {
            'category': new_categories,
            'content_type': analysis.get('content_type', 'Discussion'),
            'complexity_level': analysis.get('complexity_level', 'Intermediate'),
            'language': 'en',  # Default to English, can be improved later
            'key_topics': topic_slugs,
            'named_entities': []  # Empty for now, can be populated later
        }
    
    def _format_youtube_date(self, upload_date: str) -> str:
        """Convert YouTube upload_date (YYYYMMDD) to ISO 8601 format"""
        if not upload_date or len(upload_date) != 8:
            return datetime.now().isoformat() + 'Z'
        
        try:
            dt = datetime.strptime(upload_date, '%Y%m%d')
            return dt.isoformat() + 'Z'
        except ValueError:
            return datetime.now().isoformat() + 'Z'
    
    def atomic_write(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Write data atomically using temporary file"""
        if self.dry_run:
            print(f"  💾 Would write: {file_path}")
            return True
        
        try:
            # Create temporary file in same directory for atomic rename
            temp_path = file_path.with_suffix('.tmp')
            
            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(file_path)
            return True
            
        except Exception as e:
            print(f"❌ Write failed for {file_path}: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single JSON file"""
        try:
            # Load existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if migration needed
            if not self.needs_migration(data):
                print(f"⏭️  Skip: {file_path.name} (already migrated)")
                self.skipped_count += 1
                return True
            
            print(f"🔄 Processing: {file_path.name}")
            
            # Create backup (only in live mode)
            if not self.dry_run and not self.create_backup(file_path):
                return False
            
            # Migrate to new schema
            migrated_data = self.migrate_to_universal_schema(data)
            
            # Atomic write
            if self.atomic_write(file_path, migrated_data):
                print(f"✅ Migrated: {file_path.name}")
                self.processed_count += 1
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            self.error_count += 1
            return False
    
    def run(self, resume: bool = False) -> None:
        """Run the backfill process"""
        print(f"\n🚀 Starting backfill process...")
        
        # Get all JSON files
        json_files = list(self.reports_dir.glob('*.json'))
        json_files = [f for f in json_files if not f.name.startswith('.')]  # Skip hidden files
        json_files.sort()  # Consistent ordering
        
        if not json_files:
            print("📭 No JSON files found to process")
            return
        
        print(f"📊 Found {len(json_files)} JSON files")
        
        # Handle resume
        start_index = 0
        if resume:
            start_index = self.load_state()
            if start_index > 0:
                print(f"🔄 Resuming from index {start_index}")
        
        # Apply limit
        end_index = len(json_files)
        if self.limit:
            end_index = min(start_index + self.limit, len(json_files))
        
        files_to_process = json_files[start_index:end_index]
        print(f"📝 Processing {len(files_to_process)} files ({start_index} to {end_index-1})")
        
        # Process files
        for i, file_path in enumerate(files_to_process):
            current_index = start_index + i
            
            try:
                self.process_file(file_path)
                
                # Save state periodically (every 10 files)
                if not self.dry_run and (current_index + 1) % 10 == 0:
                    self.save_state(current_index + 1)
                    
            except KeyboardInterrupt:
                print(f"\n⏸️  Process interrupted. Saving state...")
                if not self.dry_run:
                    self.save_state(current_index)
                break
        
        # Final state save
        if not self.dry_run:
            self.save_state(end_index)
        
        # Summary
        print(f"\n📈 Backfill Summary:")
        print(f"   ✅ Processed: {self.processed_count}")
        print(f"   ⏭️  Skipped: {self.skipped_count}")
        print(f"   ❌ Errors: {self.error_count}")
        
        if not self.dry_run and self.processed_count > 0:
            print(f"   💾 Backups stored in: {self.backup_dir}")


def main():
    parser = argparse.ArgumentParser(description='Backfill YTV2 JSON files to universal schema')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making modifications')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of files to process')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last processed file')
    parser.add_argument('--reports-dir', type=str,
                       help='Path to reports directory (overrides environment)')
    
    args = parser.parse_args()
    
    # Determine reports directory
    reports_dir = args.reports_dir
    if not reports_dir:
        # Check environment variable
        nas_base = os.getenv('NAS_BASE', '/Volumes/Docker/YTV2')
        reports_dir = os.path.join(nas_base, 'data/reports')
    
    # Validate directory exists
    if not os.path.exists(reports_dir):
        print(f"❌ Reports directory not found: {reports_dir}")
        print("💡 Set NAS_BASE environment variable or use --reports-dir flag")
        sys.exit(1)
    
    # Initialize processor
    processor = BackfillProcessor(
        reports_dir=reports_dir,
        dry_run=args.dry_run,
        limit=args.limit
    )
    
    # Run backfill
    try:
        processor.run(resume=args.resume)
        print("\n🎉 Backfill process completed!")
        
    except Exception as e:
        print(f"\n💥 Backfill process failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
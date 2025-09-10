#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract video IDs from broken universal schema JSON files on NAS
Creates a list for local processing
"""

import json
import os
import sys
from datetime import datetime

# Simple Path class for older Python versions
class Path(object):
    def __init__(self, path):
        self._path = str(path)
    
    def __str__(self):
        return self._path
    
    def __div__(self, other):
        return Path(os.path.join(self._path, str(other)))
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    @property
    def name(self):
        return os.path.basename(self._path)
    
    def exists(self):
        return os.path.exists(self._path)
    
    def glob(self, pattern):
        import glob as glob_module
        full_pattern = os.path.join(self._path, pattern)
        return [Path(p) for p in glob_module.glob(full_pattern)]


def extract_video_ids_from_broken_files(reports_dir):
    """Extract video IDs from files that need duration fixes"""
    
    reports_path = Path(reports_dir)
    json_files = list(reports_path.glob('*.json'))
    
    print("ANALYZING: {} JSON files".format(len(json_files)))
    
    broken_files = []
    
    for json_file in json_files:
        try:
            with open(str(json_file), 'r') as f:
                data = json.load(f)
            
            # Only process universal schema files with duration_seconds: 0
            if ('content_source' in data and 
                data.get('content_source') == 'youtube' and 
                data.get('duration_seconds', 0) == 0):
                
                # Extract video ID
                video_id = None
                
                # Try source_metadata first
                if 'source_metadata' in data and 'youtube' in data['source_metadata']:
                    video_id = data['source_metadata']['youtube'].get('video_id')
                
                # Fallback to top-level id
                if not video_id and 'id' in data:
                    id_field = data['id']
                    if id_field.startswith('yt:'):
                        video_id = id_field.replace('yt:', '')
                    elif len(id_field) == 11:
                        video_id = id_field
                
                if video_id and video_id not in ['', 'unknown', 'error']:
                    broken_files.append({
                        'filename': json_file.name,
                        'video_id': video_id,
                        'title': data.get('title', 'Unknown'),
                        'current_duration': data.get('duration_seconds', 0)
                    })
                    
        except Exception as e:
            print("ERROR: reading {}: {}".format(json_file.name, e))
            continue
    
    print("FOUND: {} files needing duration fixes".format(len(broken_files)))
    
    # Save to CSV for easy processing
    csv_file = reports_path / 'video_ids_to_fix.csv'
    with open(str(csv_file), 'w') as f:
        f.write('filename,video_id,title,current_duration\n')
        for item in broken_files:
            # Escape commas in title
            title = item['title'].replace(',', ';').replace('"', "'")
            f.write('{},{},{},{}\n'.format(
                item['filename'],
                item['video_id'], 
                title,
                item['current_duration']
            ))
    
    print("SAVED: video_ids_to_fix.csv with {} entries".format(len(broken_files)))
    
    # Also save as JSON for easier local processing
    json_file = reports_path / 'video_ids_to_fix.json'
    with open(str(json_file), 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_files': len(broken_files),
            'files': broken_files
        }, f, indent=2)
    
    print("SAVED: video_ids_to_fix.json")
    
    return broken_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract video IDs from broken JSON files")
    parser.add_argument("--reports-dir", default="/volume1/Docker/YTV2/data/reports",
                        help="Path to reports directory")
    
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print("ERROR: Reports directory not found: {}".format(reports_dir))
        sys.exit(1)
    
    broken_files = extract_video_ids_from_broken_files(str(reports_dir))
    
    print("\nSUMMARY:")
    print("- Found {} broken files".format(len(broken_files)))
    print("- Saved video_ids_to_fix.csv")
    print("- Saved video_ids_to_fix.json") 
    print("\nNext: Download these files and run local duration fetcher")


if __name__ == "__main__":
    main()
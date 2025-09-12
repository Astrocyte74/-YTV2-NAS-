#!/usr/bin/env python3
"""
Inspect the JWST JSON file to see its structure
"""

import json
from pathlib import Path

def main():
    json_file = Path('data/reports/jwst_broke_physics_new_discoveries_challenge_cosmo_wFGUMbXwdwY.json')
    
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return
        
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("📄 JSON file structure:")
        print("=" * 50)
        
        # Show top-level keys
        print("Top-level keys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {type(value).__name__} (length: {len(value)})")
            elif isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} (size: {len(value)})")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        
        # Look for video ID in different possible fields
        possible_id_fields = ['video_id', 'id', 'canonical_url', 'webpage_url', 'url']
        print("Looking for video ID:")
        for field in possible_id_fields:
            if field in data:
                print(f"  {field}: {data[field]}")
        
        # Extract video ID from URL if present
        url_field = data.get('canonical_url', '') or data.get('webpage_url', '') or data.get('url', '')
        if url_field and 'youtube.com' in url_field:
            if 'v=' in url_field:
                video_id = url_field.split('v=')[1].split('&')[0]
                print(f"  Extracted from URL: {video_id}")
            elif 'youtu.be/' in url_field:
                video_id = url_field.split('youtu.be/')[-1].split('?')[0]
                print(f"  Extracted from short URL: {video_id}")
        
        print("\n" + "=" * 50)
        print("Title:", data.get('title', 'Not found'))
        
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")

if __name__ == "__main__":
    main()
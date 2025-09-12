#!/usr/bin/env python3
"""
JSON Data Analysis Script
Analyzes all JSON files to understand format variations and data quality issues.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import sys

def analyze_json_file(file_path):
    """Analyze a single JSON file and extract key information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            'file': file_path.name,
            'size_kb': round(file_path.stat().st_size / 1024, 2),
            'has_title': bool(data.get('title', '').strip()),
            'title_length': len(data.get('title', '')),
            'has_canonical_url': bool(data.get('canonical_url', '').strip()),
            'has_thumbnail': bool(data.get('thumbnail_url', '').strip()),
            'published_at': data.get('published_at', ''),
            'duration_seconds': data.get('duration_seconds', 0),
            'word_count': data.get('word_count', 0),
            'has_audio': data.get('media', {}).get('has_audio', False),
            'has_transcript': data.get('media', {}).get('has_transcript', False),
            'video_id': '',
            'channel_name': '',
            'view_count': 0,
            'has_analysis': False,
            'category': [],
            'content_type': '',
            'complexity': '',
            'language': '',
            'format_type': 'unknown'
        }
        
        # Extract video ID from various sources
        if 'source_metadata' in data and 'youtube' in data['source_metadata']:
            yt_meta = data['source_metadata']['youtube']
            analysis['video_id'] = yt_meta.get('video_id', '')
            analysis['channel_name'] = yt_meta.get('channel_name', 'Unknown')
            analysis['view_count'] = yt_meta.get('view_count', 0)
        
        # Check for analysis data
        if 'analysis' in data:
            analysis['has_analysis'] = True
            analysis_data = data['analysis']
            analysis['category'] = analysis_data.get('category', [])
            analysis['content_type'] = analysis_data.get('content_type', '')
            analysis['complexity'] = analysis_data.get('complexity_level', '')
            analysis['language'] = analysis_data.get('language', '')
        
        # Determine format type
        if analysis['has_title'] and analysis['has_canonical_url'] and analysis['has_analysis']:
            analysis['format_type'] = 'complete'
        elif analysis['has_title'] and analysis['has_canonical_url']:
            analysis['format_type'] = 'partial'
        elif not analysis['has_title'] and not analysis['has_canonical_url']:
            analysis['format_type'] = 'empty'
        else:
            analysis['format_type'] = 'mixed'
            
        return analysis
        
    except Exception as e:
        return {
            'file': file_path.name,
            'error': str(e),
            'format_type': 'error'
        }

def generate_report(analyses):
    """Generate comprehensive analysis report."""
    print("=" * 80)
    print("YTV2 JSON DATA ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Overall stats
    total_files = len(analyses)
    successful = [a for a in analyses if 'error' not in a]
    errors = [a for a in analyses if 'error' in a]
    
    print(f"📊 OVERVIEW")
    print(f"Total JSON files found: {total_files}")
    print(f"Successfully analyzed: {len(successful)}")
    print(f"Errors encountered: {len(errors)}")
    print()
    
    if errors:
        print("❌ ERROR FILES:")
        for error in errors:
            print(f"   - {error['file']}: {error['error']}")
        print()
    
    if not successful:
        print("No files could be analyzed successfully.")
        return
    
    # Format type breakdown
    format_counts = Counter(a['format_type'] for a in successful)
    print(f"📁 FORMAT BREAKDOWN")
    for format_type, count in format_counts.most_common():
        percentage = (count / len(successful)) * 100
        print(f"   {format_type.capitalize()}: {count} files ({percentage:.1f}%)")
    print()
    
    # Data quality analysis
    has_title = sum(1 for a in successful if a['has_title'])
    has_url = sum(1 for a in successful if a['has_canonical_url'])
    has_thumb = sum(1 for a in successful if a['has_thumbnail'])
    has_analysis = sum(1 for a in successful if a['has_analysis'])
    has_audio = sum(1 for a in successful if a['has_audio'])
    has_transcript = sum(1 for a in successful if a['has_transcript'])
    
    print(f"📈 DATA QUALITY")
    print(f"   Files with titles: {has_title}/{len(successful)} ({(has_title/len(successful)*100):.1f}%)")
    print(f"   Files with URLs: {has_url}/{len(successful)} ({(has_url/len(successful)*100):.1f}%)")
    print(f"   Files with thumbnails: {has_thumb}/{len(successful)} ({(has_thumb/len(successful)*100):.1f}%)")
    print(f"   Files with analysis: {has_analysis}/{len(successful)} ({(has_analysis/len(successful)*100):.1f}%)")
    print(f"   Files with audio: {has_audio}/{len(successful)} ({(has_audio/len(successful)*100):.1f}%)")
    print(f"   Files with transcripts: {has_transcript}/{len(successful)} ({(has_transcript/len(successful)*100):.1f}%)")
    print()
    
    # Duration analysis
    durations = [a['duration_seconds'] for a in successful if a['duration_seconds'] > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"⏱️  DURATION ANALYSIS")
        print(f"   Files with duration: {len(durations)}/{len(successful)}")
        print(f"   Average duration: {avg_duration:.0f} seconds ({avg_duration/60:.1f} minutes)")
        print(f"   Shortest: {min(durations)} seconds")
        print(f"   Longest: {max(durations)} seconds")
        print()
    
    # Published date analysis
    pub_dates = [a['published_at'] for a in successful if a['published_at']]
    date_counter = Counter(a['published_at'][:10] for a in successful if a['published_at'])
    print(f"📅 PUBLISHED DATE ANALYSIS")
    print(f"   Files with dates: {len(pub_dates)}/{len(successful)}")
    if date_counter:
        print("   Most common dates:")
        for date, count in date_counter.most_common(5):
            print(f"      {date}: {count} files")
    print()
    
    # Channel analysis
    channels = [a['channel_name'] for a in successful if a['channel_name'] and a['channel_name'] != 'Unknown']
    channel_counter = Counter(channels)
    print(f"📺 CHANNEL ANALYSIS")
    print(f"   Unique channels: {len(channel_counter)}")
    if channel_counter:
        print("   Top channels:")
        for channel, count in channel_counter.most_common(5):
            print(f"      {channel}: {count} videos")
    print()
    
    # Content analysis
    all_categories = []
    all_content_types = []
    all_complexities = []
    
    for a in successful:
        if a['has_analysis']:
            all_categories.extend(a['category'])
            if a['content_type']:
                all_content_types.append(a['content_type'])
            if a['complexity']:
                all_complexities.append(a['complexity'])
    
    print(f"🏷️  CONTENT ANALYSIS")
    
    if all_categories:
        cat_counter = Counter(all_categories)
        print(f"   Categories ({len(cat_counter)} unique):")
        for cat, count in cat_counter.most_common(5):
            print(f"      {cat}: {count} videos")
    
    if all_content_types:
        type_counter = Counter(all_content_types)
        print(f"   Content Types ({len(type_counter)} unique):")
        for ctype, count in type_counter.most_common(5):
            print(f"      {ctype}: {count} videos")
    
    if all_complexities:
        comp_counter = Counter(all_complexities)
        print(f"   Complexity Levels ({len(comp_counter)} unique):")
        for comp, count in comp_counter.most_common():
            print(f"      {comp}: {count} videos")
    print()
    
    # Recommendations
    print("🚀 RECOMMENDATIONS")
    
    incomplete_files = len(successful) - has_title
    if incomplete_files > 0:
        print(f"   • {incomplete_files} files need title/metadata enrichment")
    
    missing_analysis = len(successful) - has_analysis
    if missing_analysis > 0:
        print(f"   • {missing_analysis} files need analysis data generation")
    
    if len(format_counts) > 1:
        print(f"   • Multiple JSON formats detected - standardization needed")
    
    print(f"   • SQLite migration recommended for better performance")
    print(f"   • Estimated {len(successful)} records for database import")
    print()

def save_detailed_report(analyses, output_file):
    """Save detailed analysis to JSON file."""
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_files': len(analyses),
        'analyses': analyses
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed report saved to: {output_file}")

def main():
    # Set path to NAS data directory
    data_dir = Path("/Users/markdarby/projects/YTV_temp_NAS_files/data/reports")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure the NAS data is available at the expected path.")
        return 1
    
    print(f"🔍 Analyzing JSON files in: {data_dir}")
    print()
    
    # Find all JSON files
    json_files = list(data_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith('.')]
    
    if not json_files:
        print("❌ No JSON files found in the data directory.")
        return 1
    
    print(f"Found {len(json_files)} JSON files to analyze...")
    
    # Analyze each file
    analyses = []
    for i, json_file in enumerate(json_files, 1):
        if i % 10 == 0 or i == len(json_files):
            print(f"Progress: {i}/{len(json_files)} files analyzed")
        
        analysis = analyze_json_file(json_file)
        analyses.append(analysis)
    
    print()
    
    # Generate report
    generate_report(analyses)
    
    # Save detailed report
    output_file = Path("json_analysis_report.json")
    save_detailed_report(analyses, output_file)
    
    return 0

if __name__ == "__main__":
    exit(main())
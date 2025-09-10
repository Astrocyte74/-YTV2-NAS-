"""
JSON Report Generator Module for YTV2 YouTube Summarizer

This module handles generating, managing, and storing JSON reports from video summaries.
It provides a clean, standardized approach to report generation with proper file management
and integration with the existing YouTube processing pipeline.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import glob


class JSONReportGenerator:
    """
    Generates and manages JSON reports for YouTube video summaries.
    
    Features:
    - Standardized JSON schema for consistency
    - Clean filename conventions with conflict resolution
    - Batch report generation support
    - Report discovery and listing
    - Integration with existing YouTubeSummarizer output
    """
    
    def __init__(self, reports_dir: str = "data/reports"):
        """
        Initialize the JSON Report Generator.
        
        Args:
            reports_dir: Directory to store JSON reports (default: data/reports)
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Report schema version for future compatibility
        self.schema_version = "1.0.0"
    
    def generate_report(self, 
                       video_data: Dict[str, Any],
                       summary_data: Dict[str, Any],
                       processing_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a standardized JSON report from video and summary data.
        
        Args:
            video_data: Video metadata (title, channel, duration, etc.)
            summary_data: Summary content and analysis results
            processing_info: Processing metadata (model, provider, settings)
        
        Returns:
            Dictionary containing the complete JSON report
        """
        timestamp = datetime.now().isoformat()
        
        # Create standardized report structure
        report = {
            "metadata": {
                "schema_version": self.schema_version,
                "generated_at": timestamp,
                "report_id": self._generate_report_id(video_data, timestamp)
            },
            "video": self._extract_video_info(video_data),
            "summary": self._extract_summary_info(summary_data),
            "processing": processing_info or {},
            "stats": self._calculate_stats(video_data, summary_data)
        }
        
        return report
    
    def save_report(self, 
                   report: Dict[str, Any],
                   filename: Optional[str] = None,
                   overwrite: bool = False) -> str:
        """
        Save a JSON report to disk with proper filename handling.
        
        Args:
            report: The report dictionary to save
            filename: Optional custom filename (without extension)
            overwrite: Whether to overwrite existing files
        
        Returns:
            Absolute path to the saved report file
        """
        if not filename:
            filename = self._generate_filename(report)
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.reports_dir / filename
        
        # Handle file conflicts
        if filepath.exists() and not overwrite:
            filepath = self._resolve_filename_conflict(filepath)
        
        # Save the report
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(filepath.absolute())
    
    def generate_and_save(self,
                         video_data: Dict[str, Any],
                         summary_data: Dict[str, Any],
                         processing_info: Optional[Dict[str, Any]] = None,
                         filename: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate and save a report in one operation.
        
        Returns:
            Tuple of (report_dict, filepath)
        """
        report = self.generate_report(video_data, summary_data, processing_info)
        filepath = self.save_report(report, filename)
        return report, filepath
    
    def list_reports(self, 
                    pattern: str = "*.json",
                    sort_by: str = "date") -> List[Dict[str, Any]]:
        """
        List available reports with metadata.
        
        Args:
            pattern: File pattern to match (default: *.json)
            sort_by: Sort criteria ("date", "title", "filename")
        
        Returns:
            List of report metadata dictionaries
        """
        report_files = list(self.reports_dir.glob(pattern))
        reports = []
        
        for filepath in report_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                # Parse timestamp for display
                generated_at = report.get("metadata", {}).get("generated_at", "")
                try:
                    if generated_at:
                        dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                        created_date = dt.strftime('%Y-%m-%d')
                        created_time = dt.strftime('%H:%M')
                        timestamp = dt.isoformat()
                    else:
                        # Fallback to file modification time
                        dt = datetime.fromtimestamp(filepath.stat().st_mtime)
                        created_date = dt.strftime('%Y-%m-%d')
                        created_time = dt.strftime('%H:%M')
                        timestamp = dt.isoformat()
                except (ValueError, AttributeError):
                    # Final fallback
                    dt = datetime.fromtimestamp(filepath.stat().st_mtime)
                    created_date = dt.strftime('%Y-%m-%d')
                    created_time = dt.strftime('%H:%M')
                    timestamp = dt.isoformat()
                
                metadata = {
                    "filename": filepath.name,
                    "filepath": str(filepath.absolute()),
                    "size": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                    "title": report.get("video", {}).get("title", "Unknown"),
                    "channel": report.get("video", {}).get("channel", "Unknown"),
                    "duration": report.get("video", {}).get("duration", 0),
                    "thumbnail": report.get("video", {}).get("thumbnail", ""),
                    "url": report.get("video", {}).get("url", ""),
                    "video_id": report.get("video", {}).get("video_id", ""),
                    "generated_at": generated_at,
                    "created_date": created_date,
                    "created_time": created_time,
                    "timestamp": timestamp,
                    "model": report.get("processing", {}).get("model", "Unknown"),
                    "summary_preview": (report.get("summary", {}).get("content", "")[:150] + "...") if len(report.get("summary", {}).get("content", "")) > 150 else report.get("summary", {}).get("content", ""),
                    "report_id": report.get("metadata", {}).get("report_id", "")
                }
                reports.append(metadata)
                
            except (json.JSONDecodeError, KeyError) as e:
                # Skip invalid files
                continue
        
        # Sort reports
        if sort_by == "date":
            reports.sort(key=lambda x: x["generated_at"], reverse=True)
        elif sort_by == "title":
            reports.sort(key=lambda x: x["title"].lower())
        elif sort_by == "filename":
            reports.sort(key=lambda x: x["filename"])
        
        return reports
    
    def _extract_video_info(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize video information."""
        return {
            "url": video_data.get("url", ""),
            "video_id": video_data.get("id", ""),
            "title": video_data.get("title", ""),
            "channel": video_data.get("uploader", "") or video_data.get("channel", ""),
            "channel_id": video_data.get("uploader_id", "") or video_data.get("channel_id", ""),
            "duration": video_data.get("duration", 0),
            "duration_string": video_data.get("duration_string", ""),
            "view_count": video_data.get("view_count", 0),
            "like_count": video_data.get("like_count", 0),
            "upload_date": video_data.get("upload_date", ""),
            "description": video_data.get("description", ""),
            "tags": video_data.get("tags", []),
            "categories": video_data.get("categories", []),
            "thumbnail": video_data.get("thumbnail", ""),
            "language": video_data.get("language", ""),
            "subtitles_available": bool(video_data.get("subtitles", {}))
        }
    
    def _extract_summary_info(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize summary information."""
        return {
            "content": summary_data.get("summary", ""),
            "type": summary_data.get("summary_type", "comprehensive"),
            "analysis": summary_data.get("analysis", {}),
            "key_points": summary_data.get("key_points", []),
            "topics": summary_data.get("topics", []),
            "sentiment": summary_data.get("sentiment", {}),
            "quality_score": summary_data.get("quality_score", 0),
            "word_count": len(str(summary_data.get("summary", "")).split()) if summary_data.get("summary") else 0
        }
    
    def _calculate_stats(self, video_data: Dict[str, Any], summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate report statistics."""
        video_duration = video_data.get("duration", 0)
        summary_text = str(summary_data.get("summary", ""))
        
        return {
            "video_length_seconds": video_duration,
            "video_length_minutes": round(video_duration / 60, 1) if video_duration else 0,
            "summary_word_count": len(summary_text.split()) if summary_text else 0,
            "summary_character_count": len(summary_text),
            "compression_ratio": round(len(summary_text) / max(len(video_data.get("description", "") or ""), 1), 3),
            "has_analysis": bool(summary_data.get("analysis")),
            "has_key_points": bool(summary_data.get("key_points")),
            "topic_count": len(summary_data.get("topics", []))
        }
    
    def _generate_filename(self, report: Dict[str, Any]) -> str:
        """Generate a clean filename for the report."""
        video_info = report.get("video", {})
        title = video_info.get("title", "unknown_video")
        video_id = video_info.get("video_id", "")
        
        # Clean title for filename
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        clean_title = clean_title.strip('_').lower()
        
        # Limit length
        if len(clean_title) > 50:
            clean_title = clean_title[:50]
        
        # Add video ID if available for uniqueness
        if video_id:
            filename = f"{clean_title}_{video_id}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_title}_{timestamp}"
        
        return filename
    
    def _resolve_filename_conflict(self, filepath: Path) -> Path:
        """Canonical naming: overwrite existing files instead of creating _1, _2 suffixes"""
        return filepath
    
    def _generate_report_id(self, video_data: Dict[str, Any], timestamp: str) -> str:
        """Generate a unique report ID."""
        video_id = video_data.get("id", "")
        url = video_data.get("url", "")
        
        # Use video ID if available, otherwise hash the URL
        if video_id:
            base_id = video_id
        else:
            base_id = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Add timestamp hash for uniqueness
        time_hash = hashlib.md5(timestamp.encode()).hexdigest()[:6]
        return f"{base_id}_{time_hash}"


def create_report_from_youtube_summarizer(summarizer_result: Dict[str, Any], 
                                        processing_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper function to create a report from YouTubeSummarizer output.
    
    Args:
        summarizer_result: Output from YouTubeSummarizer (universal schema format)
        processing_info: Optional processing metadata
    
    Returns:
        Generated report dictionary in universal schema format
    """
    # Check if summarizer_result already has universal schema structure
    if 'content_source' in summarizer_result and summarizer_result.get('content_source') == 'youtube':
        # Modern universal schema format - pass through directly with minimal processing
        # Just ensure we have the required metadata structure
        if 'metadata' not in summarizer_result:
            summarizer_result['metadata'] = {
                'schema_version': '1.1.0',
                'generated_at': datetime.now().isoformat(),
                'report_id': f"{summarizer_result.get('source_metadata', {}).get('youtube', {}).get('video_id', 'unknown')}_{hash(str(summarizer_result)) % 100000:06d}"
            }
        
        return summarizer_result
    
    # Legacy compatibility - convert old format to new
    generator = JSONReportGenerator()
    
    # Extract video and summary data from summarizer result
    video_data = summarizer_result.get("metadata", {})
    summary_data = {
        "summary": summarizer_result.get("summary", ""),
        "analysis": summarizer_result.get("analysis", {}),
        "summary_type": summarizer_result.get("summary_type", "comprehensive")
    }
    
    # Extract processing info from summarizer result
    processing_info = summarizer_result.get("processor_info", processing_info)
    
    return generator.generate_report(video_data, summary_data, processing_info)


# Export the main class and convenience function
__all__ = ['JSONReportGenerator', 'create_report_from_youtube_summarizer']
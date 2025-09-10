"""
JSON Report Generator Module for YTV2 YouTube Summarizer

This module handles generating, managing, and storing JSON reports from video summaries.
It provides a clean, standardized approach to report generation with proper file management
and integration with the existing YouTube processing pipeline.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import glob


def get_mp3_duration_seconds(path: str) -> Union[int, None]:
    """
    Return duration in whole seconds using ffprobe.
    Uses robust binary detection and clean CSV output.
    """
    import shutil
    
    print(f"🔧 Extracting MP3 duration from: {path}")
    
    # Check if ffprobe is available
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print("❌ ffprobe not found in PATH")
        return None
    
    try:
        # Use CSV output for clean parsing (no JSON overhead)
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                duration_float = float(result.stdout.strip())
                duration_int = int(round(duration_float))
                print(f"✅ Extracted MP3 duration using ffprobe: {duration_int}s")
                return duration_int
            except ValueError as e:
                print(f"❌ Could not parse ffprobe duration output: {result.stdout.strip()}")
                return None
        else:
            print(f"❌ ffprobe failed: return code {result.returncode}")
            if result.stderr:
                print(f"❌ ffprobe stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ ffprobe timeout")
        return None
    except Exception as e:
        print(f"❌ ffprobe error: {type(e).__name__}: {e}")
        return None


def compute_summary_word_count(report: dict) -> int:
    """Compute word count from summary content with fallbacks for various schemas"""
    # prefer explicit counts
    for k in ("word_count",
              ("summary", "word_count"),
              ("stats", "summary_word_count"),
              ("analysis", "summary_word_count")):
        if isinstance(k, tuple):
            d = report
            ok = True
            for part in k:
                d = d.get(part, {})
                if not isinstance(d, (dict, int)):
                    ok = False
                    break
            if ok and isinstance(d, int):
                return d
        elif isinstance(report.get(k), int):
            return report[k]

    # fallback: longest summary-like text
    candidates = []
    s = report.get("summary")
    if isinstance(s, str): 
        candidates.append(s)
    if isinstance(s, dict):
        c = s.get("content")
        if isinstance(c, str): 
            candidates.append(c)
        elif isinstance(c, dict):
            for key in ("summary", "audio", "comprehensive"):
                v = c.get(key)
                if isinstance(v, str): 
                    candidates.append(v)
    a = report.get("analysis") or {}
    if isinstance(a.get("summary"), str): 
        candidates.append(a["summary"])

    if not candidates: 
        return 0
    text = max(candidates, key=len).strip()
    return len(text.split())


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
            "stats": self._calculate_stats(video_data, summary_data),
            "media_metadata": self._extract_media_metadata(video_data, summary_data)
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
    
    def _extract_media_metadata(self, video_data: Dict[str, Any], summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract media metadata for dashboard consumption."""
        video_duration = video_data.get("duration", 0)
        summary_type = summary_data.get("summary_type", "comprehensive")
        
        # Calculate word count using robust function
        # Create temp report structure for word count extraction
        temp_report = {
            "summary": summary_data,
            "word_count": summary_data.get("word_count"),
            "stats": {"summary_word_count": summary_data.get("word_count")},
            "analysis": {"summary_word_count": summary_data.get("word_count")}
        }
        word_count = compute_summary_word_count(temp_report)
        reading_time_minutes = max(1, round(word_count / 200)) if word_count > 0 else 0
        
        # Determine languages
        original_language = video_data.get("language", "en")
        summary_language = original_language
        
        # Extract language from summary type (e.g., "audio-fr" -> "fr")
        if "-" in summary_type:
            parts = summary_type.split("-")
            if len(parts) > 1 and len(parts[1]) == 2:  # Valid language code
                summary_language = parts[1]
        
        return {
            "video_duration_seconds": video_duration,
            "mp3_duration_seconds": None,  # Will be updated after TTS generation
            "mp3_voice": None,            # Will be updated after TTS generation
            "mp3_created_at": None,       # Will be updated after TTS generation
            "summary_word_count": word_count,
            "estimated_reading_time_minutes": reading_time_minutes,
            "summary_language": summary_language,
            "original_language": original_language,
            "summary_type": summary_type,
            "has_audio": False            # Will be updated after TTS generation
        }
    
    def _generate_filename(self, report: Dict[str, Any]) -> str:
        """Generate a clean filename for the report."""
        # Handle both universal schema and legacy schema
        if 'content_source' in report and report.get('content_source') == 'youtube':
            # Universal schema format
            title = report.get("title", "unknown_video")
            video_id = report.get('source_metadata', {}).get('youtube', {}).get('video_id', '')
        else:
            # Legacy schema format
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


def update_json_with_mp3_metadata(json_filepath: str, mp3_filepath: str, voice: str = "fable") -> bool:
    """
    Update an existing JSON report with MP3 metadata after TTS generation.
    
    Args:
        json_filepath: Path to the existing JSON report file
        mp3_filepath: Path to the generated MP3 file
        voice: TTS voice used (default: "fable")
    
    Returns:
        bool: True if update succeeded, False otherwise
    """
    try:
        from pathlib import Path
        import json
        from datetime import datetime
        
        # Get MP3 duration using ffprobe (clean, reliable)
        # Ensure we have absolute path for Docker compatibility
        mp3_path = Path(mp3_filepath)
        if not mp3_path.is_absolute():
            mp3_path = Path.cwd() / mp3_path
        print(f"🔍 Attempting to extract duration from: {mp3_path}")
        print(f"🔍 File exists: {mp3_path.exists()}")
        mp3_duration_seconds = get_mp3_duration_seconds(str(mp3_path))
        
        # Read existing JSON
        json_path = Path(json_filepath)
        if not json_path.exists():
            print(f"Error: JSON file not found: {json_filepath}")
            return False
            
        with open(json_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Update media metadata with MP3 info
        if "media_metadata" not in report:
            print("Warning: media_metadata section not found in JSON, creating it")
            report["media_metadata"] = {}
        
        report["media_metadata"].update({
            "mp3_duration_seconds": mp3_duration_seconds,
            "mp3_voice": voice,
            "mp3_created_at": datetime.now().isoformat(),
            "has_audio": True
        })
        
        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Updated JSON with MP3 metadata: {mp3_duration_seconds}s duration")
        return True
        
    except Exception as e:
        print(f"❌ Error updating JSON with MP3 metadata: {e}")
        return False


# Export the main class and convenience functions
__all__ = ['JSONReportGenerator', 'create_report_from_youtube_summarizer', 'update_json_with_mp3_metadata']
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

# SQLite backend support
try:
    from .sqlite_content_index import SQLiteContentIndex
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Feature flag to control SQLite usage. Default to disabled when POSTGRES_ONLY is set.
_postgres_only = os.getenv('POSTGRES_ONLY', 'false').lower() == 'true'
SQLITE_SYNC_ENABLED = os.getenv('SQLITE_SYNC_ENABLED')
if SQLITE_SYNC_ENABLED is None:
    SQLITE_SYNC_ENABLED = not _postgres_only
else:
    SQLITE_SYNC_ENABLED = SQLITE_SYNC_ENABLED.lower() == 'true'

# API client support for direct Render sync
try:
    from .render_api_client import RenderAPIClient, create_client_from_env
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

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
        
        # Initialize SQLite backend if available and enabled
        self.sqlite_db = None
        if SQLITE_AVAILABLE and SQLITE_SYNC_ENABLED:
            try:
                # Use single consolidated database location
                db_path = Path("data/ytv2_content.db")
                
                if db_path.exists():
                    self.sqlite_db = SQLiteContentIndex(str(db_path))
                    print(f"✅ SQLite backend initialized: {db_path}")
                else:
                    print(f"❌ SQLite database not found: {db_path}")
            except Exception as e:
                print(f"❌ SQLite initialization failed: {e}")
                self.sqlite_db = None
        
        # Initialize API client for direct Render sync
        self.api_client = None
        if API_CLIENT_AVAILABLE and os.getenv('RENDER_API_URL') and os.getenv('SYNC_SECRET'):
            try:
                self.api_client = create_client_from_env()
                print(f"✅ Render API client initialized")
            except Exception as e:
                print(f"❌ API client initialization failed: {e}")
                self.api_client = None
        
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
        
        # Save to both SQLite (primary) and JSON (for dual-sync compatibility)
        if self.sqlite_db:
            try:
                self._save_to_sqlite(report)
                print(f"✅ Saved to SQLite database")
            except Exception as e:
                print(f"❌ SQLite save failed: {e}")
                print(f"📊 SQLite database available ({self.sqlite_db.get_report_count()} existing records)")

        # Always create JSON file for dual-sync compatibility
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON report created: {filepath}")
        
        return str(filepath.absolute())
    
    def _save_to_sqlite(self, report: Dict[str, Any]):
        """Convert JSON report format to SQLite and save to database."""
        if not self.sqlite_db:
            return
        
        if report.get('content_source') and report.get('content_source') != 'youtube':
            # SQLite schema is tuned for YouTube ingest; skip other sources.
            print("ℹ️ Skipping SQLite save for non-YouTube content")
            return
        
        try:
            # Extract data for SQLite using same structure as our manual script
            video_id = report.get('id', '')
            if not video_id:
                print("❌ No video ID in report for SQLite")
                return
                
            # Extract video ID without prefix
            short_video_id = video_id.replace('yt:', '') if video_id.startswith('yt:') else video_id
            
            # Extract metadata
            metadata = report.get('metadata', {})
            analysis = report.get('analysis', {})
            
            # Handle summary format
            summary_data = report.get('summary', {})
            if isinstance(summary_data, dict):
                summary_text = summary_data.get('text', '') or summary_data.get('summary', '') or str(summary_data)
            else:
                summary_text = str(summary_data)
            
            # Process structured categories to generate subcategories_json and a legacy subcategory fallback
            subcategories_json = None
            legacy_subcategory = analysis.get('subcategory')
            structured_categories = analysis.get('categories')
            if not isinstance(structured_categories, list):
                structured_categories = []
            if structured_categories:
                try:
                    subcategories_json = json.dumps({"categories": structured_categories}, ensure_ascii=False)
                    if not legacy_subcategory:
                        for category_entry in structured_categories:
                            if isinstance(category_entry, dict):
                                subcats = category_entry.get('subcategories')
                                if isinstance(subcats, list) and subcats:
                                    legacy_subcategory = subcats[0]
                                    break
                except Exception as e:
                    print(f"⚠️ Error processing structured categories: {e}")
            
            # Normalise category list for backwards compatibility and JSON storage
            raw_categories = analysis.get('category') or []
            if not isinstance(raw_categories, list):
                raw_categories = [raw_categories] if raw_categories else []
            if structured_categories:
                for category_entry in structured_categories:
                    if isinstance(category_entry, dict):
                        name = category_entry.get('category')
                        if name and name not in raw_categories:
                            raw_categories.append(name)
            normalised_categories = [c for c in raw_categories if c]

            # Build record data
            now = datetime.now().isoformat()
            
            def _json_or_none(values: Any) -> Optional[str]:
                if not values:
                    return None
                if not isinstance(values, (list, tuple)):
                    values = [values]
                values = [v for v in values if v]
                return json.dumps(values, ensure_ascii=False) if values else None

            if not SQLITE_SYNC_ENABLED or not self.sqlite_db:
                return

            record_data = {
                'id': video_id,
                'title': report.get('title', 'Untitled'),
                'canonical_url': report.get('canonical_url', ''),
                'thumbnail_url': report.get('thumbnail_url', ''),
                'published_at': report.get('published_at', ''),
                'indexed_at': report.get('processed_at', now),
                'duration_seconds': report.get('duration_seconds', 0),
                'word_count': report.get('word_count', 0),
                'has_audio': bool(report.get('media_metadata', {}).get('mp3_file')),
                'audio_duration_seconds': report.get('media_metadata', {}).get('duration_seconds'),
                'has_transcript': bool(report.get('transcript')),
                'transcript_chars': len(report.get('transcript', '')),
                'video_id': short_video_id,
                'channel_name': (metadata.get('uploader') or metadata.get('channel') or 'Unknown'),
                'channel_id': metadata.get('channel_id', ''),
                'view_count': metadata.get('view_count'),
                'like_count': metadata.get('like_count'),
                'comment_count': metadata.get('comment_count'),
                'category': _json_or_none(normalised_categories),
                'subcategory': legacy_subcategory,
                'subcategories_json': subcategories_json,
                'content_type': analysis.get('content_type'),
                'complexity_level': analysis.get('complexity_level'),
                'language': analysis.get('language'),
                'key_topics': _json_or_none(analysis.get('key_topics')),
                'named_entities': _json_or_none(analysis.get('named_entities')),
                'format_source': 'direct_processing',
                'processing_status': 'completed',
                'created_at': now,
                'updated_at': now
            }
            
            # Insert into database
            conn = self.sqlite_db._get_connection()
            cursor = conn.cursor()
            
            insert_sql = '''
            INSERT OR REPLACE INTO content 
            (id, title, canonical_url, thumbnail_url, published_at, indexed_at, duration_seconds,
             word_count, has_audio, audio_duration_seconds, has_transcript, transcript_chars,
             video_id, channel_name, channel_id, view_count, like_count, comment_count,
             category, subcategory, subcategories_json, content_type, complexity_level, language, key_topics, named_entities,
             format_source, processing_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(insert_sql, (
                record_data['id'], record_data['title'], record_data['canonical_url'], record_data['thumbnail_url'],
                record_data['published_at'], record_data['indexed_at'], record_data['duration_seconds'],
                record_data['word_count'], record_data['has_audio'], record_data['audio_duration_seconds'],
                record_data['has_transcript'], record_data['transcript_chars'], record_data['video_id'],
                record_data['channel_name'], record_data['channel_id'], record_data['view_count'],
                record_data['like_count'], record_data['comment_count'], record_data['category'],
                record_data.get('subcategory'), record_data.get('subcategories_json'), record_data['content_type'], record_data['complexity_level'], record_data['language'],
                record_data['key_topics'], record_data['named_entities'], record_data['format_source'],
                record_data['processing_status'], record_data['created_at'], record_data['updated_at']
            ))
            
            # Also insert summary into content_summaries table (required for dashboard)
            summary_data = report.get('summary', {})
            if isinstance(summary_data, dict):
                summary_text = summary_data.get('text', '') or summary_data.get('summary', '') or str(summary_data)
            else:
                summary_text = str(summary_data)
            
            if summary_text:
                summary_insert_sql = '''
                INSERT OR REPLACE INTO content_summaries (content_id, summary_text, summary_type)
                VALUES (?, ?, ?)
                '''
                cursor.execute(summary_insert_sql, (record_data['id'], summary_text, 'audio'))
                print(f"📊 Summary saved to content_summaries table")
            
            conn.commit()
            print(f"📊 Successfully saved to SQLite: {record_data['title']}")
            
        except Exception as e:
            print(f"❌ SQLite save error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_sqlite_mp3_metadata(self, report_id: str, mp3_duration_seconds: int, voice: str):
        """Update SQLite record with MP3 metadata after audio generation."""
        if not SQLITE_SYNC_ENABLED or not self.sqlite_db:
            return

        try:
            conn = self.sqlite_db._get_connection()
            cursor = conn.cursor()
            
            # Update the MP3 metadata fields
            update_sql = '''
            UPDATE content 
            SET has_audio = ?, 
                audio_duration_seconds = ?,
                updated_at = ?
            WHERE id = ?
            '''
            
            now = datetime.now().isoformat()
            
            cursor.execute(update_sql, (True, mp3_duration_seconds, now, report_id))
            
            if cursor.rowcount > 0:
                conn.commit()
                print(f"📊 SQLite MP3 metadata updated for: {report_id}")
            else:
                print(f"⚠️  No SQLite record found to update for: {report_id}")
                
        except Exception as e:
            print(f"❌ SQLite MP3 metadata update error: {e}")
            import traceback
            traceback.print_exc()
    
    def sync_report_to_api(self, report: Dict[str, Any], audio_path: Optional[Path] = None) -> bool:
        """Sync report directly to Render API."""
        if not self.api_client:
            print("ℹ️  API client not available - skipping API sync")
            return False
            
        try:
            # Convert report to universal schema format
            content_data = self._convert_to_universal_schema(report)
            
            print(f"📡 Syncing report to Render API...")
            result = self.api_client.create_or_update_content(content_data)
            
            content_id = result.get('id')
            action = result.get('action', 'unknown')
            
            print(f"✅ API sync successful - {action}: {content_id}")
            
            # Upload audio if available
            if audio_path and audio_path.exists():
                print(f"🎵 Uploading audio file...")
                audio_result = self.api_client.upload_audio_file(audio_path, content_id)
                print(f"✅ Audio uploaded successfully")

            image_meta = content_data.get("summary_image") or {}
            image_path_value = image_meta.get("path") or image_meta.get("relative_path")
            if image_path_value:
                image_path = Path(image_path_value)
                if not image_path.is_absolute():
                    image_path = Path.cwd() / image_path
                if image_path.exists():
                    print("🖼️ Uploading summary image...")
                    self.api_client.upload_image_file(image_path, content_id)
                    print("✅ Summary image uploaded successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ API sync failed: {e}")
            return False
    
    def _convert_to_universal_schema(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Convert report to universal schema format for API."""
        
        # Extract data from report structure
        video_info = report.get('video', {})
        summary_info = report.get('summary', {})
        analysis_info = summary_info.get('analysis', {}) if isinstance(summary_info, dict) else {}
        
        # Build universal schema format
        content_data = {
            'content_source': 'youtube',
            'source_metadata': {
                'youtube': {
                    'video_id': video_info.get('video_id', ''),
                    'title': video_info.get('title', 'Untitled'),
                    'channel_name': video_info.get('channel', ''),
                    'published_at': video_info.get('upload_date', ''),
                    'duration_seconds': video_info.get('duration', 0),
                    'thumbnail_url': video_info.get('thumbnail', ''),
                    'canonical_url': video_info.get('url', '')
                }
            },
            'content_analysis': {
                'language': analysis_info.get('language', 'en'),
                'category': analysis_info.get('category', []) if isinstance(analysis_info.get('category'), list) else [],
                'content_type': analysis_info.get('content_type', ''),
                'complexity_level': analysis_info.get('complexity_level', ''),
                'key_topics': analysis_info.get('key_topics', []) if isinstance(analysis_info.get('key_topics'), list) else [],
                'named_entities': analysis_info.get('named_entities', []) if isinstance(analysis_info.get('named_entities'), list) else []
            },
            'media_info': {
                'has_audio': bool(report.get('media_metadata', {}).get('has_audio', False)),
                'audio_duration_seconds': report.get('media_metadata', {}).get('mp3_duration_seconds', 0),
                'has_transcript': True,  # Assume true for processed reports
                'transcript_chars': len(str(summary_info.get('content', '') if isinstance(summary_info, dict) else str(summary_info))),
                'word_count': report.get('stats', {}).get('summary_word_count', 0)
            },
            'processing_metadata': {
                'indexed_at': report.get('metadata', {}).get('generated_at', ''),
                'content_id': report.get('metadata', {}).get('report_id', '')
            }
        }
        
        # Add summary content
        summary_content = ''
        if isinstance(summary_info, dict):
            summary_content = summary_info.get('content', '') or summary_info.get('summary', '')
        elif isinstance(summary_info, str):
            summary_content = summary_info
        
        if summary_content:
            content_data['summary'] = {
                'content': {
                    'summary': summary_content,
                    'summary_type': summary_info.get('type', 'comprehensive') if isinstance(summary_info, dict) else 'comprehensive'
                }
            }
        
        return content_data
    
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
                
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
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
            "word_count": len(str(summary_data.get("summary", "")).split()) if summary_data.get("summary") else 0,
            "image": summary_data.get("summary_image"),
            "image_url": summary_data.get("summary_image_url")
            or (summary_data.get("summary_image") or {}).get("public_url"),
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
            "has_audio": False,           # Will be updated after TTS generation
            "summary_image_url": summary_data.get("summary_image_url")
            or (summary_data.get("summary_image") or {}).get("public_url"),
        }
    
    def _generate_filename(self, report: Dict[str, Any]) -> str:
        """Generate a clean filename for the report."""
        # Handle both universal schema and legacy schema
        if 'content_source' in report:
            title = report.get("title", "unknown_content")
            content_id = report.get("id", "")
            if report.get('content_source') == 'youtube':
                video_id = report.get('source_metadata', {}).get('youtube', {}).get('video_id', '')
            else:
                video_id = content_id.split(':', 1)[-1] if content_id else ''
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
    if 'content_source' in summarizer_result:
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
    Update SQLite database with MP3 metadata after TTS generation.
    
    Args:
        json_filepath: Path to identify the video (used to extract video ID)
        mp3_filepath: Path to the generated MP3 file
        voice: TTS voice used (default: "fable")
    
    Returns:
        bool: True if update succeeded, False otherwise
    """
    try:
        from pathlib import Path
        import json
        
        # Get MP3 duration using ffprobe (clean, reliable)
        # Ensure we have absolute path for Docker compatibility
        mp3_path = Path(mp3_filepath)
        if not mp3_path.is_absolute():
            mp3_path = Path.cwd() / mp3_path
        print(f"🔍 Attempting to extract duration from: {mp3_path}")
        print(f"🔍 File exists: {mp3_path.exists()}")
        mp3_duration_seconds = get_mp3_duration_seconds(str(mp3_path))
        
        # Extract video ID from file path to identify the record
        json_path = Path(json_filepath)
        video_id_match = None
        
        # Try to extract video ID from filename patterns
        filename = json_path.stem
        # Look for YouTube video ID patterns (11 characters)
        import re
        video_id_patterns = [
            r'yt_([A-Za-z0-9_-]{11})',  # yt_VIDEO_ID format
            r'([A-Za-z0-9_-]{11})',     # Direct video ID
        ]
        
        for pattern in video_id_patterns:
            match = re.search(pattern, filename)
            if match:
                video_id_match = f"yt:{match.group(1)}"
                break
        
        if not video_id_match:
            print(f"❌ Could not extract video ID from filename: {filename}")
            return False
            
        print(f"📊 Extracted video ID for SQLite update: {video_id_match}")
        
        # Initialize SQLite database connection
        try:
            from modules.sqlite_content_index import SQLiteContentIndex
            sqlite_db = SQLiteContentIndex()
            print(f"📊 SQLite database initialized")
        except Exception as e:
            print(f"❌ Failed to initialize SQLite database: {e}")
            return False
        
        # Update SQLite database directly (JSON-free workflow)
        try:
            # Use the same method logic but adapted for direct calls
            conn = sqlite_db._get_connection()
            cursor = conn.cursor()
            
            # Update the MP3 metadata fields
            update_sql = '''
            UPDATE content 
            SET has_audio = ?, 
                audio_duration_seconds = ?,
                updated_at = ?
            WHERE id = ?
            '''
            
            now = datetime.now().isoformat()
            cursor.execute(update_sql, (True, mp3_duration_seconds, now, video_id_match))
            
            if cursor.rowcount > 0:
                conn.commit()
                print(f"✅ SQLite MP3 metadata updated for: {video_id_match} ({mp3_duration_seconds}s)")
            else:
                print(f"⚠️  No SQLite record found to update for: {video_id_match}")
                
        except Exception as e:
            print(f"❌ SQLite MP3 metadata update error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating JSON with MP3 metadata: {e}")
        return False


# Export the main class and convenience functions
__all__ = ['JSONReportGenerator', 'create_report_from_youtube_summarizer', 'update_json_with_mp3_metadata']

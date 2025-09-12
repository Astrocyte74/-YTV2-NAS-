#!/usr/bin/env python3
"""
SQLite Content Index
High-performance content management using SQLite database.
Replaces the JSON-based content_index.py with proper database queries.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

class SQLiteContentIndex:
    """SQLite-based content management for YTV2 Dashboard."""
    
    def __init__(self, db_path: str = "ytv2_content.db"):
        """Initialize with database path."""
        self.db_path = Path(db_path)
        self._ensure_database_exists()
        
    def _ensure_database_exists(self):
        """Ensure the database file exists."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        
        logger.info(f"Using SQLite database: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def _parse_json_field(self, value: str) -> List[str]:
        """Parse JSON field safely."""
        if not value:
            return []
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except:
            return []
    
    def _format_report_for_api(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to API format."""
        return {
            'id': row['id'],
            'title': row['title'] or 'Untitled',
            'thumbnail_url': row['thumbnail_url'] or '',
            'canonical_url': row['canonical_url'] or '',
            'channel': row['channel_name'] or '',
            'published_at': row['published_at'] or '',
            'duration_seconds': row['duration_seconds'] or 0,
            'analysis': {
                'category': self._parse_json_field(row['category']),
                'content_type': row['content_type'] or '',
                'complexity_level': row['complexity_level'] or '',
                'language': row['language'] or 'en',
                'key_topics': self._parse_json_field(row['key_topics']),
                'named_entities': self._parse_json_field(row['named_entities'])
            },
            'media': {
                'has_audio': bool(row['has_audio']),
                'audio_duration_seconds': row['audio_duration_seconds'] or 0,
                'has_transcript': bool(row['has_transcript']),
                'transcript_chars': row['transcript_chars'] or 0
            },
            'media_metadata': {
                'video_duration_seconds': row['duration_seconds'] or 0,
                'mp3_duration_seconds': row['audio_duration_seconds'] or 0
            },
            'file_stem': self._generate_file_stem(row['video_id'], row['title']),
            'video_id': row['video_id'] or '',
            'indexed_at': row['indexed_at'] or '',
            'original_language': row['language'] or 'en',
            'summary_language': row['language'] or 'en', 
            'audio_language': row['language'] or 'en',
            'word_count': row['word_count'] or 0
        }
    
    def _generate_file_stem(self, video_id: str, title: str) -> str:
        """Generate file stem for compatibility."""
        if video_id:
            return f"{video_id}"
        # Fallback to title-based stem
        safe_title = ''.join(c for c in title.lower() if c.isalnum() or c in '-_')[:50]
        return safe_title or 'unknown'
    
    def get_reports(self, 
                   page: int = 1, 
                   size: int = 12, 
                   sort: str = 'added_desc',
                   filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get paginated and filtered reports."""
        
        conn = self._get_connection()
        try:
            # Build base query
            query = "SELECT * FROM content"
            params = []
            where_conditions = []
            
            # Apply filters
            if filters:
                if 'source' in filters and filters['source']:
                    # For now, assume all content is YouTube
                    pass
                
                if 'language' in filters and filters['language']:
                    lang_conditions = []
                    for lang in filters['language']:
                        lang_conditions.append("language = ?")
                        params.append(lang)
                    if lang_conditions:
                        where_conditions.append(f"({' OR '.join(lang_conditions)})")
                
                if 'category' in filters and filters['category']:
                    cat_conditions = []
                    for cat in filters['category']:
                        cat_conditions.append("category LIKE ?")
                        params.append(f'%"{cat}"%')
                    if cat_conditions:
                        where_conditions.append(f"({' OR '.join(cat_conditions)})")
                
                if 'content_type' in filters and filters['content_type']:
                    type_conditions = []
                    for ctype in filters['content_type']:
                        type_conditions.append("content_type = ?")
                        params.append(ctype)
                    if type_conditions:
                        where_conditions.append(f"({' OR '.join(type_conditions)})")
                
                if 'complexity' in filters and filters['complexity']:
                    comp_conditions = []
                    for comp in filters['complexity']:
                        comp_conditions.append("complexity_level = ?")
                        params.append(comp)
                    if comp_conditions:
                        where_conditions.append(f"({' OR '.join(comp_conditions)})")
                
                if 'has_audio' in filters:
                    where_conditions.append("has_audio = ?")
                    params.append(filters['has_audio'])
            
            # Add WHERE clause if we have conditions
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            # Add sorting
            sort_mapping = {
                'added_desc': 'indexed_at DESC',
                'added_asc': 'indexed_at ASC',
                'video_newest': 'published_at DESC',
                'video_oldest': 'published_at ASC',
                'title_az': 'title ASC',
                'title_za': 'title DESC',
                'duration_desc': 'duration_seconds DESC',
                'duration_asc': 'duration_seconds ASC',
                # Legacy compatibility
                'newest': 'published_at DESC',
                'oldest': 'published_at ASC',
                'title': 'title ASC',
                'title_asc': 'title ASC',
                'title_desc': 'title DESC',
                'duration': 'duration_seconds DESC'
            }
            
            order_by = sort_mapping.get(sort, 'indexed_at DESC')
            query += f" ORDER BY {order_by}"
            
            # Get total count for pagination
            count_query = query.replace("SELECT *", "SELECT COUNT(*)")
            if " ORDER BY " in count_query:
                count_query = count_query.split(" ORDER BY ")[0]
            
            cursor = conn.execute(count_query, params)
            total_count = cursor.fetchone()[0]
            
            # Add pagination
            offset = (page - 1) * size
            query += " LIMIT ? OFFSET ?"
            params.extend([size, offset])
            
            # Execute main query
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to API format
            reports = [self._format_report_for_api(row) for row in rows]
            
            # Calculate pagination info
            total_pages = (total_count + size - 1) // size
            
            return {
                'reports': reports,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                },
                'sort': sort,
                'filters': filters or {}
            }
            
        finally:
            conn.close()
    
    def get_filters(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available filter options with counts."""
        conn = self._get_connection()
        try:
            filters = {}
            
            # Source filter (always YouTube for now)
            filters['source'] = [{'value': 'youtube', 'count': self.get_report_count()}]
            
            # Language filter
            cursor = conn.execute("""
                SELECT language, COUNT(*) as count 
                FROM content 
                WHERE language IS NOT NULL AND language != ''
                GROUP BY language 
                ORDER BY count DESC
            """)
            filters['language'] = [
                {'value': row['language'], 'count': row['count']} 
                for row in cursor.fetchall()
            ]
            
            # Category filter (parse JSON arrays)
            cursor = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM content 
                WHERE category IS NOT NULL AND category != '' AND category != '[]'
                GROUP BY category
                ORDER BY count DESC
            """)
            
            category_counts = {}
            for row in cursor.fetchall():
                categories = self._parse_json_field(row['category'])
                for cat in categories:
                    category_counts[cat] = category_counts.get(cat, 0) + row['count']
            
            filters['category'] = [
                {'value': cat, 'count': count}
                for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # Content type filter
            cursor = conn.execute("""
                SELECT content_type, COUNT(*) as count
                FROM content 
                WHERE content_type IS NOT NULL AND content_type != ''
                GROUP BY content_type 
                ORDER BY count DESC
            """)
            filters['content_type'] = [
                {'value': row['content_type'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
            
            # Complexity filter
            cursor = conn.execute("""
                SELECT complexity_level, COUNT(*) as count
                FROM content 
                WHERE complexity_level IS NOT NULL AND complexity_level != ''
                GROUP BY complexity_level 
                ORDER BY count DESC
            """)
            filters['complexity_level'] = [
                {'value': row['complexity_level'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
            
            # Has audio filter
            cursor = conn.execute("""
                SELECT has_audio, COUNT(*) as count
                FROM content 
                GROUP BY has_audio
            """)
            filters['has_audio'] = [
                {'value': bool(row['has_audio']), 'count': row['count']}
                for row in cursor.fetchall()
            ]
            
            return filters
            
        finally:
            conn.close()
    
    def get_report_count(self) -> int:
        """Get total number of reports."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM content")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM content WHERE id = ?", (report_id,))
            row = cursor.fetchone()
            
            if row:
                # Get summary if available
                summary_cursor = conn.execute(
                    "SELECT summary_text, summary_type FROM content_summaries WHERE content_id = ?", 
                    (report_id,)
                )
                summary_row = summary_cursor.fetchone()
                
                report = self._format_report_for_api(row)
                
                # Add summary data
                if summary_row:
                    report['summary'] = {
                        'content': {
                            'summary': summary_row['summary_text'],
                            'summary_type': summary_row['summary_type']
                        }
                    }
                
                return report
            
            return None
            
        finally:
            conn.close()
    
    def search_reports(self, 
                      filters: Optional[Dict[str, Any]] = None,
                      query: Optional[str] = None,
                      sort: str = 'added_desc',
                      page: int = 1,
                      size: int = 20) -> Dict[str, Any]:
        """Search and filter reports with pagination - matches original API signature."""
        
        # For search queries, use get_reports with filters instead
        if query and len(query.strip()) >= 2:
            # Simple search implementation - could be enhanced later
            conn = self._get_connection()
            try:
                search_term = f"%{query.strip()}%"
                
                # Search in titles and summaries
                cursor = conn.execute("""
                    SELECT c.* FROM content c
                    LEFT JOIN content_summaries s ON c.id = s.content_id
                    WHERE c.title LIKE ? OR s.summary_text LIKE ?
                    ORDER BY 
                        CASE WHEN c.title LIKE ? THEN 1 ELSE 2 END,
                        c.indexed_at DESC
                    LIMIT ?
                """, (search_term, search_term, search_term, min(size, 50)))
                
                rows = cursor.fetchall()
                reports = [self._format_report_for_api(row) for row in rows]
                
                return {
                    'reports': reports,
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total_count': len(reports),
                        'total_pages': 1,
                        'has_next': False,
                        'has_prev': False
                    },
                    'sort': sort,
                    'filters': filters or {}
                }
                
            finally:
                conn.close()
        else:
            # No query, use standard get_reports
            return self.get_reports(page=page, size=size, sort=sort, filters=filters)
    
    def needs_refresh(self) -> bool:
        """SQLite doesn't need refresh like file-based system."""
        return False
    
    def refresh_index(self):
        """No-op for SQLite - data is always current."""
        pass
    
    def force_refresh(self) -> int:
        """Force refresh and return count - for compatibility with JSON version."""
        return self.get_report_count()
    
    def get_facets(self, active_filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get facets/filters - alias for get_filters() for compatibility."""
        return self.get_filters()
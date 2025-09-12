#!/usr/bin/env python3
"""
Render API Client
Handles communication with the YTV2 Dashboard API on Render.
Replaces database file syncing with direct API calls.
"""

import requests
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class RenderAPIClient:
    """Client for communicating with YTV2 Dashboard API on Render."""
    
    def __init__(self, base_url: str = None, auth_token: str = None):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for the Render API (e.g., https://ytv2-dashboard.onrender.com)
            auth_token: Bearer token for authentication
        """
        self.base_url = base_url or os.getenv('RENDER_API_URL', '')
        self.auth_token = auth_token or os.getenv('SYNC_SECRET', '')
        self.session = requests.Session()
        
        if not self.base_url:
            raise ValueError("RENDER_API_URL environment variable is required")
        if not self.auth_token:
            raise ValueError("SYNC_SECRET environment variable is required")
            
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'YTV2-NAS-Client/1.0'
        })
        
        logger.info(f"Initialized Render API client for {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., '/api/content')
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                # Log request details
                logger.debug(f"{method} {url} -> {response.status_code}")
                
                if response.status_code < 500:
                    # Don't retry client errors (4xx)
                    return response
                    
                # Server error (5xx) - retry
                if attempt < max_retries - 1:
                    logger.warning(f"Server error {response.status_code}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                    
                return response
                
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error: {e}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise
        
        return response
    
    def create_or_update_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update content using UPSERT logic.
        
        Args:
            content_data: Content data in universal schema format
            
        Returns:
            API response data
            
        Raises:
            requests.RequestException: If API call fails
        """
        try:
            response = self._make_request('POST', '/api/content', json=content_data)
            
            if response.status_code == 200:
                result = response.json()
                action = result.get('action', 'unknown')
                content_id = result.get('id', 'unknown')
                logger.info(f"Content {action}: {content_id}")
                return result
            else:
                error_msg = f"Failed to create/update content: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail.get('error', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                response.raise_for_status()
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def update_content(self, content_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific fields of existing content.
        
        Args:
            content_id: Content ID to update
            updates: Fields to update
            
        Returns:
            API response data
        """
        try:
            response = self._make_request('PUT', f'/api/content/{content_id}', json=updates)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Updated content: {content_id}")
                return result
            else:
                error_msg = f"Failed to update content {content_id}: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail.get('error', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                response.raise_for_status()
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def upload_audio_file(self, audio_path: Path, content_id: str) -> Dict[str, Any]:
        """Upload MP3 audio file for content.
        
        Args:
            audio_path: Path to MP3 file
            content_id: Associated content ID
            
        Returns:
            API response data
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Prepare multipart form data
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audio': (audio_path.name, audio_file, 'audio/mpeg')
                }
                data = {
                    'content_id': content_id
                }
                
                # Remove Content-Type header for multipart uploads
                headers = dict(self.session.headers)
                if 'Content-Type' in headers:
                    del headers['Content-Type']
                
                response = self._make_request(
                    'POST', 
                    '/api/upload-audio', 
                    files=files, 
                    data=data,
                    headers=headers
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Uploaded audio for content: {content_id}")
                return result
            else:
                error_msg = f"Failed to upload audio: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail.get('error', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                response.raise_for_status()
                
        except requests.RequestException as e:
            logger.error(f"Audio upload failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to the API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple GET request to check connectivity
            response = self._make_request('GET', '/api/reports', params={'size': 1})
            
            if response.status_code == 200:
                logger.info("API connection test successful")
                return True
            else:
                logger.warning(f"API connection test failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def sync_content_from_database(self, db_path: Path) -> Dict[str, Any]:
        """Sync all content from local SQLite database to Render API.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Sync statistics
        """
        import sqlite3
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        stats = {
            'processed': 0,
            'created': 0,
            'updated': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Get all content with summaries
            cursor = conn.execute("""
                SELECT c.*, s.summary_text, s.summary_type
                FROM content c
                LEFT JOIN content_summaries s ON c.id = s.content_id
                ORDER BY c.indexed_at DESC
            """)
            
            for row in cursor.fetchall():
                try:
                    # Convert database row to API format
                    content_data = self._convert_db_row_to_api_format(row)
                    
                    # Send to API
                    result = self.create_or_update_content(content_data)
                    
                    stats['processed'] += 1
                    if result.get('action') == 'created':
                        stats['created'] += 1
                    elif result.get('action') == 'updated':
                        stats['updated'] += 1
                    
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to sync content {row['id']}: {e}")
                    stats['errors'] += 1
                    continue
            
            stats['end_time'] = datetime.now().isoformat()
            logger.info(f"Sync completed: {stats}")
            return stats
            
        finally:
            conn.close()
    
    def _convert_db_row_to_api_format(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to API format."""
        import json
        
        def parse_json_field(value: str) -> List[str]:
            if not value:
                return []
            try:
                result = json.loads(value)
                return result if isinstance(result, list) else []
            except:
                return []
        
        # Build content data in universal schema format
        content_data = {
            'content_source': 'youtube',
            'source_metadata': {
                'youtube': {
                    'video_id': row['video_id'] or '',
                    'title': row['title'] or 'Untitled',
                    'channel_name': row['channel_name'] or '',
                    'published_at': row['published_at'] or '',
                    'duration_seconds': row['duration_seconds'] or 0,
                    'thumbnail_url': row['thumbnail_url'] or '',
                    'canonical_url': row['canonical_url'] or ''
                }
            },
            'content_analysis': {
                'language': row['language'] or 'en',
                'category': parse_json_field(row['category']),
                'content_type': row['content_type'] or '',
                'complexity_level': row['complexity_level'] or '',
                'key_topics': parse_json_field(row['key_topics']),
                'named_entities': parse_json_field(row['named_entities'])
            },
            'media_info': {
                'has_audio': bool(row['has_audio']),
                'audio_duration_seconds': row['audio_duration_seconds'] or 0,
                'has_transcript': bool(row['has_transcript']),
                'transcript_chars': row['transcript_chars'] or 0,
                'word_count': row['word_count'] or 0
            },
            'processing_metadata': {
                'indexed_at': row['indexed_at'] or '',
                'content_id': row['id'] or ''
            }
        }
        
        # Add summary if available
        if row['summary_text']:
            content_data['summary'] = {
                'content': {
                    'summary': row['summary_text'],
                    'summary_type': row['summary_type'] or 'comprehensive'
                }
            }
        
        return content_data


def create_client_from_env() -> RenderAPIClient:
    """Create API client using environment variables."""
    return RenderAPIClient()


# Export main class and convenience function
__all__ = ['RenderAPIClient', 'create_client_from_env']
#!/usr/bin/env python3
"""
PostgreSQL Sync Client for YTV2 NAS
Handles communication with PostgreSQL ingest endpoints on the dashboard.

This client sends content and audio to the new /ingest/* endpoints
that were implemented in T-Y020C.
"""

import os
import time
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


def find_audio_for_video(video_id: str) -> Optional[Path]:
    """
    Locate the newest MP3 produced for this video_id.
    Searches typical NAS locations and patterns:
    - ./exports/audio_<video_id>_*.mp3   (current TTS output)
    - ./exports/*<video_id>*.mp3         (catch-all)
    - ./data/exports/*<video_id>*.mp3    (legacy/backfill)
    """
    # Clean video_id (remove yt: prefix if present)
    clean_video_id = video_id.replace('yt:', '').strip()

    candidates = []
    roots = [Path("./exports"), Path("./data/exports")]
    patterns = [
        f"audio_{clean_video_id}_*.mp3",
        f"*{clean_video_id}*.mp3",
    ]

    for root in roots:
        if root.is_dir():
            for pattern in patterns:
                candidates.extend(list(root.glob(pattern)))

    if not candidates:
        logger.warning(f"No audio files found for video_id: {video_id} (clean: {clean_video_id})")
        return None

    # Pick newest by modification time
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.debug(f"Found audio for {video_id}: {newest}")
    return newest


class PostgreSQLSyncClient:
    """Client for PostgreSQL ingest endpoints with retries and circuit breaker."""

    def __init__(self, base_url: str = None, ingest_token: str = None):
        """Initialize PostgreSQL sync client.

        Args:
            base_url: PostgreSQL dashboard URL (e.g., https://ytv2-dashboard-postgres.onrender.com)
            ingest_token: X-INGEST-TOKEN for authentication
        """
        self.base_url = base_url or os.getenv('POSTGRES_DASHBOARD_URL', '')
        self.ingest_token = ingest_token or os.getenv('INGEST_TOKEN', '')

        if not self.base_url:
            raise ValueError("POSTGRES_DASHBOARD_URL environment variable is required")
        if not self.ingest_token:
            raise ValueError("INGEST_TOKEN environment variable is required")

        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')

        # Circuit breaker state
        self.consecutive_failures = 0
        self.suspended_until = None
        self.max_failures = 5  # Flip circuit breaker after 5 failures
        self.suspension_time = 300  # 5 minutes suspension

        logger.info(f"Initialized PostgreSQL sync client for {self.base_url}")

    def is_suspended(self) -> bool:
        """Check if circuit breaker is active (PG suspended due to failures)."""
        if self.suspended_until is None:
            return False

        if datetime.now() > self.suspended_until:
            # Suspension period expired, reset
            self.suspended_until = None
            self.consecutive_failures = 0
            logger.info("🔄 PostgreSQL circuit breaker reset - resuming operations")
            return False

        return True

    def _handle_success(self):
        """Reset failure counter on successful operation."""
        if self.consecutive_failures > 0:
            logger.info(f"🟢 PostgreSQL recovered after {self.consecutive_failures} failures")
        self.consecutive_failures = 0
        self.suspended_until = None

    def _handle_failure(self, error: str):
        """Track failure and potentially activate circuit breaker."""
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_failures:
            self.suspended_until = datetime.now() + timedelta(seconds=self.suspension_time)
            logger.error(f"🔴 PostgreSQL circuit breaker activated after {self.consecutive_failures} failures")
            logger.error(f"   Suspending PG operations until {self.suspended_until.strftime('%H:%M:%S')}")
            logger.error(f"   Last error: {error}")
        else:
            logger.warning(f"⚠️  PostgreSQL failure {self.consecutive_failures}/{self.max_failures}: {error}")

    def health_check(self) -> bool:
        """Check PostgreSQL ingest health endpoint.

        Returns:
            True if healthy, False otherwise
        """
        if self.is_suspended():
            return False

        try:
            response = requests.get(
                f"{self.base_url}/health/ingest",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                required_checks = {
                    'status': 'ok',
                    'token_set': True,
                    'pg_dsn_set': True
                }

                for key, expected in required_checks.items():
                    if data.get(key) != expected:
                        logger.error(f"Health check failed: {key}={data.get(key)}, expected={expected}")
                        return False

                logger.debug("✅ PostgreSQL health check passed")
                return True
            else:
                logger.error(f"Health check failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    def _make_request_with_retry(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request with exponential backoff retry.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request arguments

        Returns:
            Response JSON or None if all retries failed
        """
        if self.is_suspended():
            logger.debug("Skipping request - PostgreSQL circuit breaker active")
            return None

        url = f"{self.base_url}{endpoint}"

        # Retry configuration per OpenAI's recommendations
        max_retries = 3
        base_delay = 1.0  # 1s, 4s, 9s pattern

        # Set default headers
        headers = kwargs.get('headers', {})
        headers['X-INGEST-TOKEN'] = self.ingest_token
        kwargs['headers'] = headers

        # Set timeouts per OpenAI's recommendations
        if 'timeout' not in kwargs:
            if endpoint.endswith('/audio'):
                kwargs['timeout'] = (5, 60)  # connect=5s, read=60s for audio
            else:
                kwargs['timeout'] = (5, 30)   # connect=5s, read=30s for reports

        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, **kwargs)

                # Success cases
                if 200 <= response.status_code < 300:
                    self._handle_success()
                    return response.json()

                # Don't retry on client errors (4xx)
                if 400 <= response.status_code < 500:
                    error_msg = f"Client error {response.status_code}: {response.text[:200]}"
                    self._handle_failure(error_msg)
                    logger.error(f"❌ {method} {endpoint} - {error_msg}")
                    return None

                # Server errors (5xx) - retry
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * ((attempt + 1) ** 2)  # 1s, 4s, 9s
                        logger.warning(f"Server error {response.status_code}, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = f"Server error {response.status_code} after {max_retries} attempts"
                        self._handle_failure(error_msg)
                        return None

            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * ((attempt + 1) ** 2)
                    logger.warning(f"Network error: {e}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    error_msg = f"Network error after {max_retries} attempts: {e}"
                    self._handle_failure(error_msg)
                    return None

        return None

    def upload_content(self, content_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Upload content to PostgreSQL /ingest/report endpoint.

        Args:
            content_data: Content payload in the format expected by PostgreSQL

        Returns:
            Response data or None if failed
        """
        logger.debug(f"📤 Uploading content to PostgreSQL: {content_data.get('video_id', 'unknown')}")

        # 🔍 DEBUG: Log payload summary (truncated for readability)
        import json
        video_id = content_data.get('video_id', 'unknown')

        # Convert NAS format to PostgreSQL ingest format
        postgres_payload = self._convert_to_postgres_format(content_data)

        # Create DEEP copy for logging to avoid modifying original data
        import copy
        log_payload = copy.deepcopy(postgres_payload)

        # Truncate long text fields in DEEP COPY ONLY
        if 'summary_text' in log_payload and len(str(log_payload['summary_text'])) > 100:
            log_payload['summary_text'] = str(log_payload['summary_text'])[:100] + "... [TRUNCATED]"
        if 'summary_html' in log_payload and len(str(log_payload['summary_html'])) > 100:
            log_payload['summary_html'] = str(log_payload['summary_html'])[:100] + "... [TRUNCATED]"
        if 'summary_variants' in log_payload:
            for variant in log_payload['summary_variants']:
                if 'text' in variant and len(variant['text']) > 100:
                    variant['text'] = variant['text'][:100] + "... [TRUNCATED]"
                if 'html' in variant and len(variant['html']) > 100:
                    variant['html'] = variant['html'][:100] + "... [TRUNCATED]"

        print(f"🔍 PostgreSQL PAYLOAD for {video_id}:")
        print(f"🔍 Basic fields: title='{postgres_payload.get('title')}', channel='{postgres_payload.get('channel_name')}', has_summaries={len(postgres_payload.get('summary_variants', []))}")
        print(f"🔍 Truncated JSON: {json.dumps(log_payload, indent=2, default=str)}")

        result = self._make_request_with_retry(
            'POST',
            '/ingest/report',
            json=postgres_payload
        )

        if result:
            logger.info(f"✅ PostgreSQL content uploaded: {content_data.get('video_id')} (upserted: {result.get('upserted')})")
        else:
            logger.error(f"❌ PostgreSQL content upload failed: {content_data.get('video_id')}")

        return result

    def upload_audio(self, video_id: str, audio_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Upload audio file to PostgreSQL /ingest/audio endpoint.

        Args:
            video_id: Video ID for the audio
            audio_path: Path to MP3 file (if None, will search for it)

        Returns:
            Response data or None if failed
        """
        # Clean video_id (remove yt: prefix if present)
        clean_video_id = video_id.replace('yt:', '').strip()

        # If no audio_path provided, try to find it
        if not audio_path:
            audio_path = find_audio_for_video(video_id)
            if not audio_path:
                logger.warning(f"Audio not found for {video_id}; skipping audio upload")
                return None

        if not audio_path.is_file():
            logger.error(f"Audio file not found or not a file: {audio_path}")
            return None

        logger.debug(f"🎵 Uploading audio to PostgreSQL: {video_id} -> {audio_path.name}")

        try:
            with open(audio_path, 'rb') as audio_file:
                # Use clean video_id for filename (no yt: prefix)
                files = {
                    'audio': (f"{clean_video_id}.mp3", audio_file, 'audio/mpeg')
                }
                data = {
                    'video_id': clean_video_id  # Send clean video_id
                }

                result = self._make_request_with_retry(
                    'POST',
                    '/ingest/audio',
                    files=files,
                    data=data,
                    headers={}  # Don't set Content-Type for multipart
                )

                if result:
                    logger.info(f"✅ PostgreSQL audio uploaded: {video_id} -> {result.get('public_url')}")
                else:
                    logger.error(f"❌ PostgreSQL audio upload failed: {video_id}")

                return result

        except Exception as e:
            logger.error(f"Audio upload error: {e}")
            self._handle_failure(str(e))
            return None

    def _convert_to_postgres_format(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NAS content format to PostgreSQL ingest format.

        The PostgreSQL ingest expects a flatter structure than the universal schema.
        """
        # Extract video_id from various possible locations
        video_id = (
            content_data.get('video_id') or
            content_data.get('id', '').replace('yt:', '') or
            content_data.get('source_metadata', {}).get('youtube', {}).get('video_id') or
            ''
        )

        # Ensure raw YouTube ID for ingest (strip yt: prefix if present)
        if video_id.startswith('yt:'):
            video_id = video_id.split('yt:', 1)[1]

        # Extract channel name from various possible locations
        channel_name = (
            content_data.get('channel_name') or
            content_data.get('source_metadata', {}).get('youtube', {}).get('channel_name') or
            content_data.get('metadata', {}).get('uploader') or
            'Unknown'
        )

        # Extract language info
        analysis_language = content_data.get('analysis', {}).get('language', 'en')
        original_language = content_data.get('original_language', analysis_language)
        summary_language = content_data.get('summary_language', analysis_language)
        audio_language = content_data.get('audio_language', summary_language)

        # Build payload matching what T-Y020C expects
        payload = {
            'id': f'yt:{video_id}',  # PostgreSQL expects yt: prefix in id field
            'video_id': video_id,    # Clean video ID
            'title': content_data.get('title', 'Untitled'),
            'channel': channel_name,      # PostgreSQL expects both channel and channel_name
            'channel_name': channel_name,
            'published_at': content_data.get('published_at'),
            'duration_seconds': content_data.get('duration_seconds'),
            'thumbnail_url': content_data.get('thumbnail_url'),
            'canonical_url': content_data.get('canonical_url'),
            'indexed_at': content_data.get('indexed_at') or datetime.now(timezone.utc).isoformat(),
            # Language fields required by PostgreSQL
            'original_language': original_language,
            'summary_language': summary_language,
            'audio_language': audio_language,
            # Word count from source data
            'word_count': content_data.get('word_count', 0)
        }

        # Add JSON fields for PostgreSQL JSONB columns
        if content_data.get('subcategories_json'):
            payload['subcategories_json'] = content_data['subcategories_json']

        if content_data.get('analysis'):
            payload['analysis_json'] = content_data['analysis']

        # Add summary data to match existing PostgreSQL format
        summary_data = content_data.get('summary', {})

        def _format_html(text: str) -> str:
            from html import escape
            if not text:
                return '<p></p>'
            paragraphs = [escape(block).replace('\n', '<br/>') for block in text.strip().split('\n\n') if block.strip()]
            if not paragraphs:
                return '<p></p>'
            return ''.join(f"<p>{paragraph}</p>" for paragraph in paragraphs)

        summary_variants = []
        processed_variants = set()
        chosen_summary_text = ''
        chosen_variant = 'comprehensive'

        if isinstance(summary_data, dict):
            # Direct summary payload (newer pipeline)
            direct_text = summary_data.get('summary')
            if isinstance(direct_text, str) and direct_text.strip():
                chosen_summary_text = direct_text.strip()
                chosen_variant = summary_data.get('summary_type', 'comprehensive')
                summary_variants.append({
                    'variant': chosen_variant,
                    'text': chosen_summary_text,
                    'html': _format_html(chosen_summary_text)
                })
                processed_variants.add(chosen_variant)

            # Multi-format payload from chunked summaries and auxiliary variants
            variant_priority = [
                'comprehensive',
                'key-insights',
                'key_points',
                'key-points',
                'key_insights',
                'executive',
                'adaptive',
                'summary'
            ]

            for key, value in summary_data.items():
                if key in {'summary', 'headline', 'summary_type', 'generated_at'}:
                    continue
                if not isinstance(value, str):
                    continue
                clean_value = value.strip()
                if not clean_value:
                    continue

                variant_name = key.replace('_', '-')
                if key == 'audio' and summary_data.get('summary_type'):
                    variant_name = summary_data['summary_type']
                if variant_name in processed_variants:
                    continue

                summary_variants.append({
                    'variant': variant_name,
                    'text': clean_value,
                    'html': _format_html(clean_value)
                })
                processed_variants.add(variant_name)

            if summary_variants and not chosen_summary_text:
                # Choose best variant based on priority order
                for preferred in variant_priority:
                    match = next((variant for variant in summary_variants if variant['variant'] == preferred), None)
                    if match:
                        chosen_summary_text = match['text']
                        chosen_variant = match['variant']
                        break
                else:
                    chosen_summary_text = summary_variants[0]['text']
                    chosen_variant = summary_variants[0]['variant']

        if chosen_summary_text:
            payload['summary_text'] = chosen_summary_text
            payload['summary_html'] = _format_html(chosen_summary_text)
            payload['summary_variant'] = chosen_variant

        if summary_variants:
            payload['summary_variants'] = summary_variants

        return payload


def create_postgres_client_from_env() -> PostgreSQLSyncClient:
    """Create PostgreSQL sync client using environment variables."""
    return PostgreSQLSyncClient()


# Export main class and convenience function
__all__ = ['PostgreSQLSyncClient', 'create_postgres_client_from_env']

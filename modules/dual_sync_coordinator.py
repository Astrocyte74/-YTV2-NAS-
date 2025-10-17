#!/usr/bin/env python3
"""
Dual-Sync Coordinator for YTV2 NAS
Coordinates synchronization to both SQLite (legacy) and PostgreSQL (new) systems.

Feature flags:
- DUAL_SYNC=true|false: Enable dual-sync mode (sends to both targets)
- POSTGRES_ONLY=true|false: Future state - sends only to PostgreSQL
- PG_INGEST_ENABLED=true|false: Quick kill-switch for PostgreSQL if it flaps

Behavior matrix:
- If DUAL_SYNC=true and one target fails → log + continue; don't block the other
- When ready to stop SQLite: set DUAL_SYNC=false and leave code path intact
- Order: POST /ingest/report before /ingest/audio (audio sets has_audio after metadata exists)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from .render_api_client import create_client_from_env as create_sqlite_client
from .postgres_writer import create_postgres_writer_from_env
from .metrics import metrics
from .event_stream import emit_report_event

logger = logging.getLogger(__name__)


class DualSyncCoordinator:
    """Coordinates synchronization to both SQLite and PostgreSQL targets."""

    def __init__(self):
        """Initialize dual-sync coordinator with feature flag detection."""
        # Feature flags from environment
        self.dual_sync_enabled = os.getenv('DUAL_SYNC', 'false').lower() == 'true'
        self.postgres_only = os.getenv('POSTGRES_ONLY', 'false').lower() == 'true'
        self.pg_ingest_enabled = os.getenv('PG_INGEST_ENABLED', 'true').lower() == 'true'
        sqlite_flag = os.getenv('SQLITE_SYNC_ENABLED')
        if sqlite_flag is None:
            self.sqlite_enabled = not self.postgres_only
        else:
            self.sqlite_enabled = sqlite_flag.lower() == 'true'

        if self.postgres_only:
            self.sqlite_enabled = False

        # Initialize clients
        self.sqlite_client = None
        self.postgres_client = None

        # Stats tracking
        self.stats = {}
        if self.sqlite_enabled:
            self.stats['sqlite'] = {'report_success': 0, 'report_fail': 0, 'audio_success': 0, 'audio_fail': 0}
        self.stats['postgres'] = {'report_success': 0, 'report_fail': 0, 'audio_success': 0, 'audio_fail': 0}

        logger.info(f"🚀 Dual-sync coordinator initialized:")
        logger.info(f"   DUAL_SYNC={self.dual_sync_enabled}")
        logger.info(f"   POSTGRES_ONLY={self.postgres_only}")
        logger.info(f"   SQLITE_SYNC_ENABLED={self.sqlite_enabled}")
        logger.info(f"   PG_INGEST_ENABLED={self.pg_ingest_enabled}")

        # Initialize clients based on configuration
        self._init_clients()

    def _init_clients(self):
        """Initialize appropriate clients based on feature flags."""
        # SQLite client (unless POSTGRES_ONLY)
        if self.sqlite_enabled:
            try:
                # Ensure RENDER_API_URL is set (fallback to RENDER_DASHBOARD_URL)
                if not os.getenv('RENDER_API_URL') and os.getenv('RENDER_DASHBOARD_URL'):
                    os.environ['RENDER_API_URL'] = os.getenv('RENDER_DASHBOARD_URL')
                    logger.info("Using RENDER_DASHBOARD_URL as RENDER_API_URL fallback")

                self.sqlite_client = create_sqlite_client()
                logger.info("✅ SQLite client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize SQLite client: {e}")
                logger.warning("⚠️  SQLite sync will be skipped")

        # PostgreSQL client (if dual-sync or postgres-only, and not disabled)
        if (self.dual_sync_enabled or self.postgres_only) and self.pg_ingest_enabled:
            try:
                self.postgres_client = create_postgres_writer_from_env()
                logger.info("✅ PostgreSQL writer initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize PostgreSQL client: {e}")
                logger.warning("⚠️  PostgreSQL sync will be skipped")

    def health_check(self) -> Dict[str, bool]:
        """Check health of all configured targets.

        Returns:
            Dictionary with health status per target
        """
        health = {}

        # SQLite health check
        if self.sqlite_client and self.sqlite_enabled:
            try:
                health['sqlite'] = self.sqlite_client.test_connection()
            except Exception as e:
                logger.error(f"SQLite health check failed: {e}")
                health['sqlite'] = False

        # PostgreSQL health check
        if self.postgres_client and self.pg_ingest_enabled:
            try:
                health['postgres'] = self.postgres_client.health_check()
            except Exception as e:
                logger.error(f"PostgreSQL health check failed: {e}")
                health['postgres'] = False

        return health

    def sync_content(self, content_data: Dict[str, Any], audio_path: Optional[Path] = None) -> Dict[str, Any]:
        """Sync content and audio to all configured targets.

        Args:
            content_data: Content data in NAS format
            audio_path: Optional path to audio file

        Returns:
            Results from all targets
        """
        video_id = content_data.get('video_id', content_data.get('id', 'unknown'))
        results: Dict[str, Dict[str, Optional[Any]]] = {}
        if self.sqlite_enabled and self.sqlite_client:
            results['sqlite'] = {'report': None, 'audio': None}
        if self.postgres_client and self.pg_ingest_enabled:
            results['postgres'] = {'report': None, 'audio': None}

        logger.info(f"🔄 Starting dual-sync for {video_id}")

        # 1) SQLite sync (existing/legacy path)
        if 'sqlite' in results:
            results['sqlite'] = self._sync_to_sqlite(content_data, audio_path, video_id)

        # 2) PostgreSQL sync (new path)
        if 'postgres' in results:
            results['postgres'] = self._sync_to_postgres(content_data, audio_path, video_id)

        # Log summary
        self._log_sync_summary(video_id, results)
        return results

    def _sync_to_sqlite(self, content_data: Dict[str, Any], audio_path: Optional[Path], video_id: str) -> Dict[str, Any]:
        """Sync to SQLite target."""
        result = {'report': None, 'audio': None}

        try:
            # Content sync
            logger.info(f"[SYNC] target=sqlite video_id={video_id} op=report status=attempting")
            report_result = self.sqlite_client.create_or_update_content(content_data)
            result['report'] = report_result
            self.stats['sqlite']['report_success'] += 1
            logger.info(f"[SYNC] target=sqlite video_id={video_id} op=report status=ok action={report_result.get('action', 'unknown')}")

            # Audio sync
            if audio_path and audio_path.is_file():
                logger.info(f"[SYNC] target=sqlite video_id={video_id} op=audio status=attempting")
                audio_result = self.sqlite_client.upload_audio_file(audio_path, video_id)
                result['audio'] = audio_result
                self.stats['sqlite']['audio_success'] += 1
                logger.info(f"[SYNC] target=sqlite video_id={video_id} op=audio status=ok")

        except Exception as e:
            error_msg = str(e)[:100]  # Truncate for logging
            logger.error(f"[SYNC] target=sqlite video_id={video_id} op=report/audio status=fail error={error_msg}")
            self.stats['sqlite']['report_fail'] += 1
            if audio_path:
                self.stats['sqlite']['audio_fail'] += 1

        return result

    @staticmethod
    def _normalize_video_id(video_id: str) -> str:
        """Normalize IDs for Postgres (strip yt: prefix when present)."""
        if not video_id:
            return video_id
        if isinstance(video_id, str) and video_id.startswith("yt:"):
            return video_id.split(":", 1)[-1]
        return video_id

    def _sync_to_postgres(self, content_data: Dict[str, Any], audio_path: Optional[Path], video_id: str) -> Dict[str, Any]:
        """Sync to PostgreSQL target with proper ordering."""
        result = {'report': None, 'audio': None}

        raw_video_id = video_id or ""
        db_video_id = self._normalize_video_id(raw_video_id)

        try:
            # Health gate before writing (per OpenAI recommendation)
            if not self.postgres_client.health_check():
                logger.warning(f"[SYNC] target=postgres video_id={raw_video_id} status=skip reason=health_check_failed")
                return result

            # Content sync FIRST (establishes metadata)
            logger.info(f"[SYNC] target=postgres video_id={raw_video_id} op=report status=attempting")
            report_result = self.postgres_client.upload_content(content_data)

            if report_result:
                result['report'] = report_result
                self.stats['postgres']['report_success'] += 1
                metrics.record_ingest(True, db_video_id)
                emit_report_event(
                    'report-synced',
                    {
                        'video_id': db_video_id,
                        'content_id': content_data.get('id') or f"yt:{db_video_id}",
                        'summary_type': (content_data.get('summary') or {}).get('summary_type'),
                        'targets': ['postgres'],
                    },
                )
                logger.info(f"[SYNC] target=postgres video_id={raw_video_id} op=report status=ok upserted={report_result.get('upserted')}")

                # Audio sync SECOND (sets has_audio flag after metadata exists)
                if audio_path and audio_path.is_file():
                    logger.info(f"[SYNC] target=postgres video_id={raw_video_id} op=audio status=attempting")
                    audio_result = self.postgres_client.upload_audio(db_video_id, audio_path)

                    if audio_result:
                        result['audio'] = audio_result
                        self.stats['postgres']['audio_success'] += 1
                        metrics.record_audio(True)
                        emit_report_event(
                            'audio-synced',
                            {
                                'video_id': db_video_id,
                                'content_id': content_data.get('id') or f"yt:{db_video_id}",
                                'audio_path': str(audio_path),
                                'targets': ['postgres'],
                            },
                        )
                        logger.info(f"[SYNC] target=postgres video_id={raw_video_id} op=audio status=ok url={audio_result.get('public_url')}")
                    else:
                        self.stats['postgres']['audio_fail'] += 1
                        metrics.record_audio(False)
                        emit_report_event(
                            'audio-sync-failed',
                            {
                                'video_id': db_video_id,
                                'content_id': content_data.get('id') or f"yt:{db_video_id}",
                                'audio_path': str(audio_path),
                            },
                        )
                        logger.error(f"[SYNC] target=postgres video_id={raw_video_id} op=audio status=fail")
            else:
                self.stats['postgres']['report_fail'] += 1
                metrics.record_ingest(False, db_video_id)
                emit_report_event(
                    'report-sync-failed',
                    {
                        'video_id': db_video_id,
                        'content_id': content_data.get('id') or f"yt:{db_video_id}",
                    },
                )
                logger.error(f"[SYNC] target=postgres video_id={raw_video_id} op=report status=fail")

        except Exception as e:
            error_msg = str(e)[:100]  # Truncate for logging
            logger.error(f"[SYNC] target=postgres video_id={raw_video_id} op=report/audio status=fail error={error_msg}")
            self.stats['postgres']['report_fail'] += 1
            metrics.record_ingest(False, db_video_id)
            emit_report_event(
                'report-sync-error',
                {
                    'video_id': db_video_id,
                    'content_id': content_data.get('id') or f"yt:{db_video_id}",
                    'error': error_msg,
                },
            )
            if audio_path:
                self.stats['postgres']['audio_fail'] += 1
                metrics.record_audio(False)
                emit_report_event(
                    'audio-sync-error',
                    {
                        'video_id': db_video_id,
                        'content_id': content_data.get('id') or f"yt:{db_video_id}",
                        'audio_path': str(audio_path),
                        'error': error_msg,
                    },
                )

        return result

    def _log_sync_summary(self, video_id: str, results: Dict[str, Any]):
        """Log a summary of sync results."""
        success_states = {target: bool(result.get('report')) for target, result in results.items()}

        if not success_states:
            logger.warning(f"⚠️  No active sync targets for {video_id}")
            return

        successful = [t for t, ok in success_states.items() if ok]
        unsuccessful = [t for t, ok in success_states.items() if not ok]

        if unsuccessful:
            if successful:
                logger.warning(
                    f"⚠️  Partial sync for {video_id}: "
                    + ", ".join(f"{t}=ok" for t in successful)
                    + "; "
                    + ", ".join(f"{t}=fail" for t in unsuccessful)
                )
            else:
                logger.error(f"❌ Dual-sync FAILED for {video_id} (no targets succeeded)")
        else:
            logger.info(f"✅ Sync SUCCESS for {video_id} ({', '.join(successful)})")

    def print_stats_summary(self):
        """Print sync statistics summary."""
        total_processed = 0
        logger.info("📊 SYNC STATISTICS SUMMARY")
        logger.info("=" * 50)

        for target, stats in self.stats.items():
            total_reports = stats['report_success'] + stats['report_fail']
            total_audio = stats['audio_success'] + stats['audio_fail']

            if total_reports > 0 or total_audio > 0:
                report_rate = (stats['report_success'] / total_reports * 100) if total_reports > 0 else 0
                audio_rate = (stats['audio_success'] / total_audio * 100) if total_audio > 0 else 0

                logger.info(f"{target.upper()}:")
                logger.info(f"  Reports: {stats['report_success']}/{total_reports} ({report_rate:.1f}%)")
                logger.info(f"  Audio:   {stats['audio_success']}/{total_audio} ({audio_rate:.1f}%)")

                total_processed = max(total_processed, total_reports)

        return total_processed

    def get_active_targets(self) -> list:
        """Get list of currently active sync targets."""
        targets = []

        if self.sqlite_enabled and self.sqlite_client:
            targets.append('sqlite')

        if (self.dual_sync_enabled or self.postgres_only) and self.postgres_client and self.pg_ingest_enabled:
            targets.append('postgres')

        return targets


# Convenience function for creating coordinator
def create_dual_sync_coordinator() -> DualSyncCoordinator:
    """Create dual-sync coordinator from environment configuration."""
    return DualSyncCoordinator()


# Export main class and convenience function
__all__ = ['DualSyncCoordinator', 'create_dual_sync_coordinator']

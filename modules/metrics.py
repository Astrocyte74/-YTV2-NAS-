#!/usr/bin/env python3
"""Lightweight metrics collector for NAS processing pipeline."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, Optional


class ProcessingMetrics:
    """Thread-safe container for ingest and runtime stats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            'ingest_success': 0,
            'ingest_failure': 0,
            'audio_success': 0,
            'audio_failure': 0,
            'tts_success': 0,
            'tts_failure': 0,
            'sse_clients_total': 0,
            'sse_events_broadcast': 0,
            'sse_events_failed': 0,
            'reprocess_requested': 0,
            'reprocess_success': 0,
            'reprocess_failure': 0,
        }
        self._sse_clients_current = 0
        self._last_ingest_at: Optional[str] = None
        self._last_ingest_video: Optional[str] = None

    def record_ingest(self, success: bool, video_id: Optional[str] = None) -> None:
        with self._lock:
            key = 'ingest_success' if success else 'ingest_failure'
            self._counters[key] += 1
            if success:
                self._last_ingest_at = datetime.utcnow().isoformat() + 'Z'
                if video_id:
                    self._last_ingest_video = video_id

    def record_audio(self, success: bool) -> None:
        with self._lock:
            key = 'audio_success' if success else 'audio_failure'
            self._counters[key] += 1

    def record_tts(self, success: bool) -> None:
        with self._lock:
            key = 'tts_success' if success else 'tts_failure'
            self._counters[key] += 1

    def record_sse_register(self) -> None:
        with self._lock:
            self._sse_clients_current += 1
            self._counters['sse_clients_total'] += 1

    def record_sse_unregister(self) -> None:
        with self._lock:
            if self._sse_clients_current > 0:
                self._sse_clients_current -= 1

    def record_event_broadcast(self, delivered: int, failed: int = 0) -> None:
        with self._lock:
            self._counters['sse_events_broadcast'] += max(delivered, 0)
            if failed:
                self._counters['sse_events_failed'] += max(failed, 0)

    def record_reprocess_request(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._counters['reprocess_requested'] += count

    def record_reprocess_result(self, success: bool) -> None:
        with self._lock:
            key = 'reprocess_success' if success else 'reprocess_failure'
            self._counters[key] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            data = {
                'counters': dict(self._counters),
                'sse_clients_current': self._sse_clients_current,
                'last_ingest_at': self._last_ingest_at,
                'last_ingest_video': self._last_ingest_video,
                'generated_at': datetime.utcnow().isoformat() + 'Z',
            }
        return data


metrics = ProcessingMetrics()

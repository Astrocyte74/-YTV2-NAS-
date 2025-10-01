#!/usr/bin/env python3
"""Server-sent events broadcaster shared across NAS services."""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from .metrics import metrics

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from http.server import SimpleHTTPRequestHandler

logger = logging.getLogger(__name__)


class SSEClient:
    """Lightweight holder for per-connection state."""

    def __init__(self, handler: Optional['SimpleHTTPRequestHandler'] = None) -> None:
        self.handler = handler
        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=64)
        self.alive = True

    def enqueue(self, message: str) -> None:
        if not self.alive:
            return
        try:
            self.queue.put_nowait(message)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(message)
            except queue.Full:
                logger.debug("SSE queue overflow; dropping event")


class ReportEventStream:
    """Thread-safe broadcaster for dashboard report events."""

    def __init__(self) -> None:
        self._clients: set[SSEClient] = set()
        self._lock = threading.Lock()

    def register(self, handler: Optional['SimpleHTTPRequestHandler'] = None) -> SSEClient:
        client = SSEClient(handler)
        with self._lock:
            self._clients.add(client)
            total = len(self._clients)
        metrics.record_sse_register()
        logger.debug("SSE client registered; total=%s", total)
        return client

    def unregister(self, client: SSEClient) -> None:
        client.alive = False
        with self._lock:
            self._clients.discard(client)
            total = len(self._clients)
        metrics.record_sse_unregister()
        logger.debug("SSE client unregistered; total=%s", total)

    def broadcast(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> int:
        if not event_name:
            return 0
        message = self._format_message(event_name, payload or {})
        stale: list[SSEClient] = []
        delivered = 0
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.enqueue(message)
                delivered += 1
            except Exception:
                logger.exception("Failed to enqueue SSE message; marking client stale")
                stale.append(client)
        for client in stale:
            self.unregister(client)
        metrics.record_event_broadcast(delivered, failed=len(stale))
        return delivered

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @staticmethod
    def _format_message(event_name: str, payload: Dict[str, Any]) -> str:
        try:
            data = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            logger.exception("Failed to serialise SSE payload for %s", event_name)
            data = json.dumps({'error': 'serialization-error'}, ensure_ascii=False)
        return f"event: {event_name}\ndata: {data}\n\n"


report_event_stream = ReportEventStream()


def emit_report_event(event_name: str, payload: Optional[Dict[str, Any]] = None) -> int:
    """Broadcast a report-related event to all SSE clients."""
    payload = dict(payload or {})
    if 'timestamp' not in payload:
        payload['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    return report_event_stream.broadcast(event_name, payload)


def active_client_count() -> int:
    """Return the number of currently connected SSE clients."""
    return report_event_stream.client_count()


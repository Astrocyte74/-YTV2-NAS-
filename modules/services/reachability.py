from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import requests


_CACHE: Dict[str, Tuple[float, bool]] = {}


def _timeouts() -> Tuple[float, float, float]:
    """Return (connect_timeout, http_timeout, ttl_seconds)."""
    try:
        ct = float(os.getenv("REACH_CONNECT_TIMEOUT", "2"))
    except Exception:
        ct = 2.0
    try:
        ht = float(os.getenv("REACH_HTTP_TIMEOUT", "4"))
    except Exception:
        ht = 4.0
    try:
        ttl = float(os.getenv("REACH_TTL_SECONDS", "20"))
    except Exception:
        ttl = 20.0
    return (ct, ht, ttl)


def _cached(key: str) -> Optional[bool]:
    ct, ht, ttl = _timeouts()
    now = time.time()
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, ok = entry
    if now - ts <= ttl:
        return ok
    return None


def _store(key: str, ok: bool) -> bool:
    _CACHE[key] = (time.time(), ok)
    return ok


def hub_ok(base_url: Optional[str] = None) -> bool:
    """Quick probe to the TTS/Ollama hub. Uses /meta endpoint.

    Returns cached status for a short TTL to avoid repeated network calls.
    """
    base = (base_url or os.getenv("TTSHUB_API_BASE") or "").strip()
    if not base:
        return False
    key = f"hub_ok:{base}"
    val = _cached(key)
    if val is not None:
        return val
    ct, ht, _ = _timeouts()
    try:
        r = requests.get(f"{base.rstrip('/')}/meta", timeout=(ct, ht))
        ok = 200 <= r.status_code < 500  # consider 4xx as reachable but misconfigured
        return _store(key, ok)
    except Exception:
        return _store(key, False)


def hub_ollama_ok(base_url: Optional[str] = None) -> bool:
    """Probe hub's Ollama proxy (ps or tags)."""
    base = (base_url or os.getenv("TTSHUB_API_BASE") or "").strip()
    if not base:
        return False
    key = f"hub_ollama_ok:{base}"
    val = _cached(key)
    if val is not None:
        return val
    ct, ht, _ = _timeouts()
    try:
        r = requests.get(f"{base.rstrip('/')}/ollama/ps", timeout=(ct, ht))
        if 200 <= r.status_code < 300:
            return _store(key, True)
        # Fallback to /tags if /ps is not supported
        r2 = requests.get(f"{base.rstrip('/')}/ollama/tags", timeout=(ct, ht))
        ok = 200 <= r2.status_code < 300
        return _store(key, ok)
    except Exception:
        return _store(key, False)


def local_ollama_ok(base_url: Optional[str] = None) -> bool:
    """Probe direct Ollama instance (bypassing hub)."""
    base = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
    key = f"local_ollama_ok:{base}"
    val = _cached(key)
    if val is not None:
        return val
    ct, ht, _ = _timeouts()
    try:
        r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=(ct, ht))
        ok = 200 <= r.status_code < 300
        return _store(key, ok)
    except Exception:
        return _store(key, False)


__all__ = ["hub_ok", "hub_ollama_ok", "local_ollama_ok"]


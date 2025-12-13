"""Filesystem-backed queue for deferred Z-Image jobs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

QUEUE_DIR = Path("data/zimage_queue")
QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9:_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "job"


def enqueue(job: Dict) -> Path:
    """Persist a Z-Image job to disk for later processing."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    chat_id = _sanitize(str(job.get("chat_id") or ""))
    prompt = _sanitize((job.get("prompt") or "")[:32])
    filename = f"pending_{ts}_{chat_id}_{prompt}.json"
    path = QUEUE_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    return path


def list_jobs() -> Dict[str, Path]:
    jobs: Dict[str, Path] = {}
    for entry in QUEUE_DIR.glob("*.json"):
        jobs[entry.stem] = entry
    return jobs


__all__ = ["enqueue", "list_jobs", "QUEUE_DIR"]

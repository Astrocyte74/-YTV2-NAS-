"""Simple filesystem-backed queue for deferred TTS jobs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

QUEUE_DIR = Path("data/tts_queue")
QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def enqueue(job: Dict) -> Path:
    """Persist a TTS job to disk for later processing."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    job_id = job.get("id") or timestamp
    filename = f"{timestamp}_{job_id}.json"
    path = QUEUE_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)
    return path


def list_jobs() -> Dict[str, Path]:
    """Return a mapping of job ids to file paths."""
    jobs = {}
    for entry in QUEUE_DIR.glob("*.json"):
        jobs[entry.stem] = entry
    return jobs


__all__ = ["enqueue", "list_jobs", "QUEUE_DIR"]

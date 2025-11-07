"""Filesystem-backed queue for deferred summary image jobs."""

from __future__ import annotations

import json
from datetime import datetime
import re
from pathlib import Path
from typing import Dict

QUEUE_DIR = Path("data/image_queue")
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9:_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def _stable_path_for_job(job: Dict) -> Path | None:
    try:
        content = job.get("content") or {}
        raw_id = str(content.get("id") or content.get("video_id") or "").strip()
        if not raw_id:
            return None
        # Namespace + key
        if raw_id.startswith("reddit:"):
            ns = "reddit"; key = raw_id.split(":",1)[-1]
        elif raw_id.startswith("yt:"):
            ns = "yt"; key = raw_id.split(":",1)[-1]
        else:
            ns = "yt" if len(raw_id) == 11 else "web"
            key = raw_id
        fname = f"pending_{ns}_{_sanitize(key)}.json"
        return QUEUE_DIR / fname
    except Exception:
        return None


def enqueue(job: Dict) -> Path:
    """Persist an image job to disk for later processing (idempotent per content id)."""
    stable = _stable_path_for_job(job)
    if stable is not None:
        try:
            with open(stable, "x", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False, indent=2)
            return stable
        except FileExistsError:
            return stable
        except Exception:
            pass
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    job_id = job.get("id") or ts
    filename = f"{ts}_{job_id}.json"
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

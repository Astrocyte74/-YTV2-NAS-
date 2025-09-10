"""
Simple JSON ledger for tracking processed YouTube videos
Prevents duplicate processing and enables re-sync capabilities
"""

import json
import os
from pathlib import Path
from contextlib import contextmanager

LEDGER_PATH = Path("data/ledger.json")
_LOCK = LEDGER_PATH.with_suffix(".lock")

@contextmanager
def _lock():
    """Simple advisory lock for single-user setup"""
    try:
        _LOCK.touch(exist_ok=True)
        yield
    finally:
        try: 
            _LOCK.unlink()
        except: 
            pass

def _load():
    """Load ledger with corruption recovery"""
    if not LEDGER_PATH.exists(): 
        return {}
    try:
        with open(LEDGER_PATH, "r") as f: 
            return json.load(f)
    except Exception:
        # Auto-backup corrupt file and start fresh
        try: 
            os.replace(LEDGER_PATH, LEDGER_PATH.with_suffix(".corrupt.json"))
            print(f"⚠️ Corrupted ledger backed up to {LEDGER_PATH.with_suffix('.corrupt.json')}")
        except: 
            pass
        return {}

def _save(d):
    """Atomic save with lock"""
    with _lock():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f: 
            json.dump(d, f, indent=2)
        os.replace(tmp, LEDGER_PATH)

def get(video_id, summary_type="short"):
    """Get ledger entry for video_id:summary_type"""
    return _load().get(f"{video_id}:{summary_type}")

def upsert(video_id, summary_type, entry):
    """Insert or update ledger entry"""
    d = _load()
    d[f"{video_id}:{summary_type}"] = entry
    _save(d)

def list_all():
    """Get all ledger entries (for debugging)"""
    return _load()
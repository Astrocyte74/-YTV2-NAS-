from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nas_sync import dual_sync_upload
from modules import ledger

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def _to_path(value: Optional[PathLike]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def run_dual_sync(
    report_path: PathLike,
    audio_path: Optional[PathLike] = None,
    *,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute dual-sync upload and return normalized results."""
    path = _to_path(report_path)
    audio = _to_path(audio_path)
    label = label or path.stem

    try:
        sync_results = dual_sync_upload(path, audio) if audio else dual_sync_upload(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("⚠️ Dual-sync error for %s: %s", label, exc)
        return {
            "success": False,
            "targets": [],
            "sync_results": None,
            "error": str(exc),
        }

    sqlite_ok = False
    postgres_ok = False
    invalid_result = False
    if isinstance(sync_results, dict):
        sqlite_ok = bool(sync_results.get("sqlite", {}).get("report"))
        postgres_ok = bool(sync_results.get("postgres", {}).get("report"))
    else:
        invalid_result = True
        logger.error("❌ Dual-sync returned unexpected result for %s: %r", label, sync_results)

    targets: List[str] = []
    if sqlite_ok:
        targets.append("SQLite")
    if postgres_ok:
        targets.append("PostgreSQL")

    success = bool(targets)
    if success:
        logger.info("✅ Dual-sync success: %s → %s", label, ", ".join(targets))
    else:
        logger.error("❌ Dual-sync failed for %s", label)

    error: Optional[str]
    if success:
        error = None
    elif invalid_result:
        error = "invalid-sync-result"
    else:
        error = "no-targets"

    return {
        "success": success,
        "targets": targets,
        "sync_results": sync_results,
        "error": error,
    }


def update_ledger_after_sync(
    video_id: str,
    summary_type: str,
    *,
    targets: Optional[List[str]],
    audio_path: Optional[PathLike] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize ledger updates after a dual-sync attempt."""
    entry = dict(ledger.get(video_id, summary_type) or {})

    if extra_fields:
        entry.update(extra_fields)

    if audio_path is not None:
        entry["mp3"] = str(_to_path(audio_path))

    entry["synced"] = bool(targets)
    entry["last_synced"] = datetime.now().isoformat()

    if targets:
        entry["sync_targets"] = targets

    ledger.upsert(video_id, summary_type, entry)
    return entry

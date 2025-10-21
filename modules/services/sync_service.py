from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nas_sync import dual_sync_upload
from modules import ledger
from modules.render_api_client import create_client_from_env as create_render_client

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]
REPORTS_DIR = Path("/app/data/reports")
_RENDER_CLIENT = None


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


def _find_report_for_video(video_id: str, preferred: Optional[Path] = None) -> Optional[Path]:
    """Locate the most recent report JSON for a given video identifier."""
    if preferred and preferred.exists():
        return preferred

    if not REPORTS_DIR.exists():
        logger.warning("⚠️ Reports directory missing: %s", REPORTS_DIR)
        return None

    matches: List[Path] = []
    for path in REPORTS_DIR.glob("*.json"):
        name = path.name
        if (
            name == f"{video_id}.json"
            or name.endswith(f"_{video_id}.json")
            or f"_{video_id}_" in name
            or (video_id in name and len(video_id) >= 8)
        ):
            matches.append(path)

    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def sync_audio_variant(
    video_id: str,
    summary_type: str,
    audio_path: PathLike,
    *,
    ledger_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync audio variant alongside existing content metadata."""
    audio = _to_path(audio_path)
    if audio is None or not audio.exists():
        logger.warning("⚠️ Audio sync skipped; file missing for %s:%s (%s)", video_id, summary_type, audio_path)
        return {"success": False, "error": "missing-audio", "targets": []}

    ledger_key = ledger_id or (f"yt:{video_id}" if video_id else video_id)
    ledger_entry = ledger.get(ledger_key, summary_type) if ledger_key else None

    preferred_json = None
    if ledger_entry:
        candidate = ledger_entry.get("json")
        if candidate:
            preferred_json = Path(candidate)

    report_path = _find_report_for_video(video_id, preferred_json)
    if not report_path:
        logger.error("❌ Could not locate JSON report for %s:%s", video_id, summary_type)
        return {"success": False, "error": "missing-report", "targets": []}

    outcome = run_dual_sync(report_path, audio, label=f"{video_id}:{summary_type}")

    if ledger_key:
        extra_fields: Dict[str, Any] = {'json': str(report_path)}
        update_ledger_after_sync(
            ledger_key,
            summary_type,
            targets=outcome["targets"],
            audio_path=audio,
            extra_fields=extra_fields,
        )

    return {
        "success": outcome["success"],
        "error": outcome.get("error"),
        "targets": outcome["targets"],
        "report_path": str(report_path),
    }


def upload_audio_to_render(content_id: str, audio_path: PathLike) -> bool:
    """Upload MP3 to Render via API client."""
    global _RENDER_CLIENT
    audio = _to_path(audio_path)
    if audio is None or not audio.exists():
        logger.warning("⚠️ Render upload skipped; file missing: %s", audio_path)
        return False

    try:
        if _RENDER_CLIENT is None:
            _RENDER_CLIENT = create_render_client()
    except Exception as exc:
        logger.warning("⚠️ Render client unavailable: %s", exc)
        return False

    try:
        _RENDER_CLIENT.upload_audio_file(audio, content_id)
        logger.info("✅ Uploaded audio to Render for %s", content_id)
        return True
    except Exception as exc:  # pragma: no cover - network call
        logger.warning("⚠️ Render audio upload failed for %s: %s", content_id, exc)
        return False

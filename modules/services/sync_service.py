from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nas_sync import dual_sync_upload
from modules import ledger

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]
REPORTS_DIR = Path("/app/data/reports")


def _to_path(value: Optional[PathLike]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _update_report_media_metadata(
    report_path: Path,
    audio_url: Optional[str],
    duration: Optional[int],
    version: Optional[int],
) -> None:
    """
    Persist audio metadata alongside the local report JSON so subsequent
    uploads keep media fields in sync with the dashboard contract.
    """
    try:
        if not report_path or not report_path.exists():
            return
        data = json.loads(report_path.read_text(encoding="utf-8"))

        media = data.get("media") or {}
        media["has_audio"] = True
        if audio_url:
            media["audio_url"] = audio_url
        data["media"] = media

        media_meta = data.get("media_metadata") or {}
        if isinstance(duration, int) and duration > 0:
            media_meta["mp3_duration_seconds"] = int(duration)
        data["media_metadata"] = media_meta

        if version is not None:
            data["audio_version"] = int(version)

        report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unable to persist audio metadata for %s: %s", report_path, exc)


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

    if audio:
        audio_meta: Optional[Dict[str, Any]] = None
        if isinstance(sync_results, dict):
            postgres_audio = (sync_results.get("postgres") or {}).get("audio") or {}
            sqlite_audio = (sync_results.get("sqlite") or {}).get("audio") or {}
            candidate = postgres_audio if postgres_audio else sqlite_audio
            if candidate and candidate.get("status") == "ok":
                audio_meta = candidate
        if audio_meta:
            raw_duration = audio_meta.get("duration")
            try:
                duration_value = int(raw_duration) if raw_duration is not None else None
            except (TypeError, ValueError):
                duration_value = None
            raw_version = audio_meta.get("audio_version")
            try:
                version_value = int(raw_version) if raw_version is not None else None
            except (TypeError, ValueError):
                version_value = None
            _update_report_media_metadata(
                path,
                audio_meta.get("audio_url"),
                duration_value,
                version_value,
            )

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

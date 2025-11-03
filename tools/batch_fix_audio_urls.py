#!/usr/bin/env python3
from __future__ import annotations

"""
Batch backfill media.audio_url + media_metadata.mp3_duration_seconds for rows
with has_audio=true but missing audio_url, using local MP3s under /app/exports.

Strategy per row:
  - Find a local MP3 by video_id: audio_{video_id}_*.mp3 (skip *_chunk*.mp3)
  - If found, call tools/fix_audio_url_for_item.py to:
      - upload to Render, derive /exports/audio/<filename>.mp3
      - update Postgres: media.audio_url, mp3_duration_seconds, audio_version, has_audio

Usage:
  python3 tools/batch_fix_audio_urls.py --limit 50           # real run
  python3 tools/batch_fix_audio_urls.py --limit 50 --dry-run # preview only

Env required:
  - DATABASE_URL
  - RENDER_DASHBOARD_URL (or RENDER_API_URL)
  - INGEST_TOKEN or SYNC_SECRET (for upload auth)
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("psycopg is required inside the container.") from exc


EXPORTS_DIR = Path("/app/exports")
FIX_TOOL = Path("/app/tools/fix_audio_url_for_item.py")


@dataclass
class Row:
    content_id: Optional[str]
    video_id: str


def discover_missing(limit: int) -> List[Row]:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    sql = (
        "SELECT id, video_id FROM content "
        "WHERE has_audio = TRUE AND ((media->>'audio_url') IS NULL OR (media->>'audio_url')='') "
        "ORDER BY updated_at DESC LIMIT %s"
    )
    rows: List[Row] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            for cid, vid in cur.fetchall():
                rows.append(Row(cid, vid))
    return rows


def find_local_mp3(video_id: str) -> Optional[Path]:
    if not EXPORTS_DIR.exists():
        return None
    candidates: List[Path] = []
    patterns = [
        f"audio_{video_id}_*.mp3",
        f"{video_id}_*.mp3",
        f"*_{video_id}_*.mp3",
    ]
    for patt in patterns:
        candidates.extend(EXPORTS_DIR.glob(patt))
    # Filter out chunked files
    candidates = [
        p for p in candidates
        if "_chunk" not in p.name and not p.name.startswith("._")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def guess_content_id(row: Row) -> str:
    if row.content_id:
        return str(row.content_id)
    vid = row.video_id
    # Heuristic: 11-char mixed-case → YouTube
    if len(vid) == 11:
        return f"yt:{vid}"
    return f"reddit:{vid}"


def run_fix(content_id: str, video_id: str, mp3_path: Path, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"DRY-RUN would fix {video_id} using {mp3_path.name}"
    if not FIX_TOOL.exists():
        return False, f"fix tool missing: {FIX_TOOL}"
    try:
        cmd = [
            sys.executable,
            str(FIX_TOOL),
            "--",  # ensure positional args even if video_id starts with '-'
            content_id,
            video_id,
            str(mp3_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ok = proc.returncode == 0
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return ok, out.strip()
    except Exception as exc:
        return False, f"exception: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = discover_missing(args.limit)
    if not rows:
        print({"checked": 0, "updated": 0, "skipped": 0})
        return 0

    updated = 0
    skipped = 0
    for row in rows:
        mp3 = find_local_mp3(row.video_id)
        if not mp3:
            skipped += 1
            print(f"skip {row.video_id}: no local mp3 found under {EXPORTS_DIR}")
            continue
        cid = guess_content_id(row)
        ok, msg = run_fix(cid, row.video_id, mp3, args.dry_run)
        print(msg)
        if ok:
            updated += 1
        else:
            skipped += 1

    print({"checked": len(rows), "updated": updated, "skipped": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

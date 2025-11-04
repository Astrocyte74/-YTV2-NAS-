#!/usr/bin/env python3
from __future__ import annotations

"""
Scan /app/exports for MP3s and backfill Postgres rows that are missing
media.audio_url and/or have has_audio=false.

This complements batch_fix_audio_urls.py by discovering candidates from files
on disk (instead of only DB rows with has_audio=true).

Usage:
  python3 tools/scan_and_fix_from_exports.py --dry-run --limit 50
  python3 tools/scan_and_fix_from_exports.py --limit 100

Environment:
  DATABASE_URL, RENDER_DASHBOARD_URL (or RENDER_API_URL), INGEST_TOKEN/SYNC_SECRET
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import requests

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("psycopg is required inside the container.") from exc


EXPORTS = Path("/app/exports")
FIX_TOOL = Path("/app/tools/fix_audio_url_for_item.py")


VID_RX = re.compile(r"^[A-Za-z0-9_-]{6,}$")


def extract_video_id_from_name(name: str) -> Optional[str]:
    # Accept patterns like audio_<video_id>_timestamp.mp3
    if name.startswith("audio_"):
        rem = name[len("audio_"):]
        vid = rem.split("_")[0]
        if VID_RX.match(vid):
            return vid
        return None
    # Accept patterns like <video_id>_timestamp.mp3 (11-char typical, but allow 6+)
    root = name.split("_")[0]
    if VID_RX.match(root):
        return root
    return None


def discover_from_exports(limit: int) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    if not EXPORTS.exists():
        return pairs
    files = [p for p in EXPORTS.glob("*.mp3") if not p.name.startswith("._") and "_chunk" not in p.name]
    # Most recent first
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        vid = extract_video_id_from_name(p.stem)
        if not vid:
            continue
        pairs.append((vid, p))
        if limit and len(pairs) >= limit:
            break
    return pairs


def select_needing_fix(dsn: str, pairs: Sequence[Tuple[str, Path]], cap: int) -> List[Tuple[str, Path, Optional[str]]]:
    need: List[Tuple[str, Path, Optional[str]]] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for vid, path in pairs:
                cur.execute(
                    (
                        "SELECT id, has_audio, (media->>'audio_url') AS audio_url "
                        "FROM content WHERE video_id = %s"
                    ),
                    (vid,),
                )
                row = cur.fetchone()
                if not row:
                    continue
                content_id, has_audio, audio_url = row
                # Fix when audio_url missing OR has_audio is false
                if (audio_url is None or audio_url == "") or (not has_audio):
                    need.append((vid, path, content_id))
                    if cap and len(need) >= cap:
                        break
    return need


def guess_content_id(content_id: Optional[str], video_id: str) -> str:
    if content_id:
        return str(content_id)
    if len(video_id) == 11:
        return f"yt:{video_id}"
    return f"reddit:{video_id}"


def run_fix(content_id: str, video_id: str, mp3: Path, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"DRY-RUN would fix {video_id} using {mp3.name}"
    if not FIX_TOOL.exists():
        return False, f"missing fix tool: {FIX_TOOL}"
    try:
        cmd = [sys.executable, str(FIX_TOOL), "--", content_id, video_id, str(mp3)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        ok = proc.returncode == 0
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return ok, out.strip()
    except Exception as exc:
        return False, f"exception: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="max files to examine from exports (default 100)")
    ap.add_argument("--cap", type=int, default=50, help="max fixes to attempt this run (default 50)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--respect-storage", action="store_true", help="Check /api/health/storage and abort when critically full")
    ap.add_argument("--block-pct", type=int, default=int(os.getenv("DASHBOARD_STORAGE_BLOCK_PCT") or 98), help="Block when used_pct >= this value (default env or 98)")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required")

    pairs = discover_from_exports(args.limit)
    if not pairs:
        print({"checked": 0, "candidates": 0, "updated": 0})
        return 0

    if args.respect_storage:
        base = (os.getenv("RENDER_DASHBOARD_URL") or os.getenv("RENDER_API_URL") or "").rstrip("/")
        tok = os.getenv("DASHBOARD_DEBUG_TOKEN")
        if base and tok:
            try:
                r = requests.get(f"{base}/api/health/storage", headers={"Authorization": f"Bearer {tok}"}, timeout=10)
                if r.status_code == 200:
                    used_pct = int((r.json() or {}).get("used_pct") or 0)
                    if used_pct >= args.block_pct:
                        print({"blocked": True, "used_pct": used_pct, "threshold": args.block_pct})
                        return 2
            except Exception:
                pass

    need = select_needing_fix(dsn, pairs, args.cap)
    if not need:
        print({"checked": len(pairs), "candidates": 0, "updated": 0})
        return 0

    updated = 0
    for vid, mp3, cid in need:
        content_id = guess_content_id(cid, vid)
        ok, msg = run_fix(content_id, vid, mp3, args.dry_run)
        print(msg)
        if ok:
            updated += 1

    print({"checked": len(pairs), "candidates": len(need), "updated": updated})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

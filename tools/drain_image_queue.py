#!/usr/bin/env python3
"""
Drain queued summary image jobs.

Reads jobs from data/image_queue, calls summary_image_service.maybe_generate_summary_image,
and uploads the saved image to the dashboard when possible. Jobs that cannot
be processed (hub offline) are left for the next run; hard failures are moved
to data/image_queue/failed.

Usage:
  python tools/drain_image_queue.py                     # process all pending jobs once
  python tools/drain_image_queue.py --limit 10          # process up to 10
  python tools/drain_image_queue.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from modules.image_queue import QUEUE_DIR


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s - drain_image_queue - %(levelname)s - %(message)s")


def _move(path: Path, outcome: str) -> None:
    dest = QUEUE_DIR / outcome
    dest.mkdir(parents=True, exist_ok=True)
    try:
        path.rename(dest / path.name)
    except Exception as exc:
        logging.warning("Could not move job %s to %s: %s", path.name, outcome, exc)


def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.error("Failed to read job %s: %s", path.name, exc)
        return None


async def process_job(path: Path) -> bool:
    from modules.services.summary_image_service import maybe_generate_summary_image
    job = _load(path)
    if not job:
        _move(path, "failed")
        return False
    content = job.get("content") or {}
    if not isinstance(content, dict):
        logging.error("Invalid job payload (no content dict): %s", path.name)
        _move(path, "failed")
        return False
    # Attempt generation; if hub offline, maybe_generate_summary_image returns None but may re-enqueue.
    meta = await maybe_generate_summary_image(content)
    if not isinstance(meta, dict):
        logging.info("Skipped (hub offline or error): %s", path.name)
        return False

    # Upload to dashboard if possible
    content_id = content.get("id") or content.get("video_id")
    image_path = Path(meta.get("path") or meta.get("relative_path") or "")
    if content_id and image_path and image_path.exists():
        try:
            from modules.render_api_client import create_client_from_env
            client = create_client_from_env()
            cid = str(content_id)
            # Normalize to the same namespace used elsewhere (e.g., yt:<video_id>)
            if len(cid) == 11 and ":" not in cid:
                cid = f"yt:{cid}"
            upload_info = client.upload_image_file(image_path, cid)
            logging.info("Uploaded image for %s", cid)

            # Prefer URL from upload_info; fallback to metadata/public_url
            summary_image_url = (
                (upload_info or {}).get("public_url")
                or (upload_info or {}).get("relative_url")
                or meta.get("public_url")
                or meta.get("relative_path")
            )

            # Best-effort DB update so the dashboard shows the image
            if summary_image_url:
                try:
                    import os as _os
                    dsn = _os.getenv("DATABASE_URL_POSTGRES_NEW") or _os.getenv("DATABASE_URL")
                    if dsn:
                        import psycopg
                        vid = str(content_id)
                        if vid.startswith("yt:") and len(vid) == 14:
                            vid = vid.split(":", 1)[-1]
                        with psycopg.connect(dsn) as _conn:
                            with _conn.cursor() as _cur:
                                _cur.execute(
                                    "UPDATE content SET summary_image_url=%s, updated_at=now() WHERE video_id=%s",
                                    (summary_image_url, vid),
                                )
                                _conn.commit()
                        logging.info("Updated DB summary_image_url for %s", vid)
                except Exception as db_exc:
                    logging.warning("DB update for summary_image_url failed: %s", db_exc)
        except Exception as exc:
            logging.warning("Render upload failed for %s: %s", content_id, exc)

    _move(path, "processed")
    return True


def drain_once(limit: Optional[int] = None) -> int:
    jobs = sorted(QUEUE_DIR.glob("*.json"))
    if limit is not None:
        jobs = jobs[: max(0, int(limit))]
    if not jobs:
        logging.info("No queued image jobs found.")
        return 0
    ok = 0
    for p in jobs:
        try:
            res = asyncio.run(process_job(p))
            ok += int(bool(res))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.error("Unhandled error: %s", exc)
            _move(p, "failed")
    return ok


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description="Drain queued summary image jobs")
    ap.add_argument("--limit", type=int, default=None, help="Max jobs to process per run")
    ap.add_argument("--watch", action="store_true", help="Keep watching and draining the queue")
    ap.add_argument("--interval", type=int, default=30, help="Polling interval seconds when --watch is set")
    args = ap.parse_args()
    if not args.watch:
        count = drain_once(args.limit)
        logging.info("Processed %d image job(s)", count)
        return 0
    logging.info("Watching image queue at %s (interval=%ss)", QUEUE_DIR, args.interval)
    try:
        import time
        while True:
            drain_once(args.limit)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

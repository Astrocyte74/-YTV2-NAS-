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
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from modules.image_queue import QUEUE_DIR
from modules.services import summary_image_service


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
    # Suppress auto-enqueue behavior inside the drainer. The drainer should
    # never create additional queue files on offline/errors; it should leave
    # the current job in place for the next pass.
    import os as _os

    prev_suppress = _os.environ.get("SUMMARY_IMAGE_QUEUE_SUPPRESS")
    prev_health = _os.environ.get("SUMMARY_IMAGE_HEALTH_BYPASS")
    _os.environ["SUMMARY_IMAGE_QUEUE_SUPPRESS"] = "1"
    # Skip repeated per-job health probes; a single preflight is done in drain_once.
    _os.environ["SUMMARY_IMAGE_HEALTH_BYPASS"] = "1"

    try:
        job = _load(path)
        if not job:
            _move(path, "failed")
            return False
        content = job.get("content") or {}
        if not isinstance(content, dict):
            logging.error("Invalid job payload (no content dict): %s", path.name)
            _move(path, "failed")
            return False
        analysis_from_db: Dict[str, Any] = {}
        try:
            vid = str(content.get("id") or content.get("video_id") or "")
            base_vid = vid.split(":", 1)[-1] if ":" in vid else vid
            dsn = os.getenv("DATABASE_URL_POSTGRES_NEW") or os.getenv("DATABASE_URL")
            if dsn and base_vid:
                import psycopg
                with psycopg.connect(dsn) as _conn:
                    with _conn.cursor() as _cur:
                        _cur.execute("SELECT summary_image_url, analysis_json FROM content WHERE video_id=%s", (base_vid,))
                        row = _cur.fetchone()
                        if row:
                            existing_url = row[0]
                            raw_analysis = row[1]
                            if raw_analysis:
                                if isinstance(raw_analysis, str):
                                    try:
                                        analysis_from_db = json.loads(raw_analysis)
                                    except Exception:
                                        analysis_from_db = {}
                                elif isinstance(raw_analysis, dict):
                                    analysis_from_db = raw_analysis
                            prompt_from_db = (analysis_from_db.get("summary_image_prompt") or "").strip()
                            prompt_last_used = (analysis_from_db.get("summary_image_prompt_last_used") or "").strip()
                            if existing_url and (not prompt_from_db or prompt_from_db == prompt_last_used):
                                logging.info("Skip job %s: summary_image_url already set", base_vid)
                                _move(path, "processed")
                                return True
                            job_analysis = content.get("analysis")
                            if isinstance(job_analysis, dict):
                                merged = dict(analysis_from_db)
                                merged.update(job_analysis)
                                if prompt_from_db:
                                    merged["summary_image_prompt"] = prompt_from_db
                                content["analysis"] = merged
                            elif isinstance(analysis_from_db, dict):
                                copy_analysis = dict(analysis_from_db)
                                if prompt_from_db:
                                    copy_analysis["summary_image_prompt"] = prompt_from_db
                                content["analysis"] = copy_analysis
        except Exception as _exc:
            logging.debug("DB check skipped: %s", _exc)

        # Attempt generation; if hub offline, maybe_generate_summary_image returns None but does not re-enqueue in drain.
        meta = await summary_image_service.maybe_generate_summary_image(content)
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

                if summary_image_url:
                    try:
                        dsn = _os.getenv("DATABASE_URL_POSTGRES_NEW") or _os.getenv("DATABASE_URL")
                        if dsn:
                            import psycopg

                            vid = str(content_id or "")
                            if ":" in vid:
                                vid = vid.split(":", 1)[-1]
                            variant_entry = meta.get("analysis_variant")
                            if not variant_entry:
                                variant_entry = summary_image_service.build_analysis_variant(meta, summary_image_url)
                            updated_analysis = summary_image_service.apply_analysis_variant(
                                analysis_from_db,
                                variant_entry,
                                selected_url=summary_image_url,
                                prompt=meta.get("prompt"),
                                model=meta.get("model"),
                            )
                            analysis_payload = json.dumps(updated_analysis)
                            with psycopg.connect(dsn) as _conn:
                                with _conn.cursor() as _cur:
                                    _cur.execute(
                                        """
                                        UPDATE content
                                           SET summary_image_url=%s,
                                               analysis_json=%s,
                                               updated_at=now()
                                         WHERE video_id=%s
                                        """,
                                        (summary_image_url, analysis_payload, vid),
                                    )
                                    _conn.commit()
                            logging.info("Updated DB summary_image_url for %s", vid)
                    except Exception as db_exc:
                        logging.warning("DB update for summary_image_url failed: %s", db_exc)
            except Exception as exc:
                logging.warning("Render upload failed for %s: %s", content_id, exc)

        _move(path, "processed")
        return True
    finally:
        if prev_suppress is None:
            _os.environ.pop("SUMMARY_IMAGE_QUEUE_SUPPRESS", None)
        else:
            _os.environ["SUMMARY_IMAGE_QUEUE_SUPPRESS"] = prev_suppress
        if prev_health is None:
            _os.environ.pop("SUMMARY_IMAGE_HEALTH_BYPASS", None)
        else:
            _os.environ["SUMMARY_IMAGE_HEALTH_BYPASS"] = prev_health


def drain_once(limit: Optional[int] = None) -> int:
    # Preflight: perform one lightweight health probe; if unreachable, skip all jobs
    try:
        import os as _os
        from modules.services import draw_service as _ds
        base = _os.getenv("TTSHUB_API_BASE") or ""
        if base:
            health = asyncio.run(_ds.fetch_drawthings_health(base, ttl=0, force_refresh=True))
            if not bool((health or {}).get("reachable", False)):
                logging.info("Image drain: hub not reachable; skipping this pass")
                return 0
    except Exception as _exc:
        logging.info("Image drain: health probe failed; skipping this pass")
        return 0

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

#!/usr/bin/env python3
"""Drain queued TTS jobs once or in a simple loop.

Reads JSON jobs from data/tts_queue/, generates audio via local TTS hub (or
OpenAI fallback if requested), and for summary jobs performs DB/Render sync.

Usage:
  python tools/drain_tts_queue.py            # process all pending jobs once
  python tools/drain_tts_queue.py --watch    # poll every N seconds (default 30)
  python tools/drain_tts_queue.py --interval 10 --watch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from modules.tts_queue import QUEUE_DIR


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - drain_tts_queue - %(levelname)s - %(message)s",
    )


def load_job(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.error(f"Failed to read job {path.name}: {exc}")
        return None


def move_job(path: Path, outcome: str) -> None:
    dest_dir = QUEUE_DIR / outcome
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.rename(dest_dir / path.name)
    except Exception as exc:
        logging.warning(f"Could not move job {path.name} to {outcome}: {exc}")


def find_report_path(video_id: str) -> Optional[Path]:
    if not video_id:
        return None
    reports_dir = Path("/app/data/reports")
    if not reports_dir.exists():
        reports_dir = Path("data/reports")
    if not reports_dir.exists():
        return None
    candidates = []
    for p in reports_dir.glob("*.json"):
        name = p.name
        if (
            name == f"{video_id}.json"
            or name.endswith(f"_{video_id}.json")
            or f"_{video_id}_" in name
            or (video_id in name and len(video_id) >= 8)
        ):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


async def process_job(path: Path) -> bool:
    job = load_job(path)
    if not job:
        move_job(path, "failed")
        return False

    mode = job.get("mode") or "summary_audio"
    text = job.get("summary_text") or job.get("text") or ""
    if not text:
        logging.error(f"Job {path.name} has no text; skipping")
        move_job(path, "failed")
        return False

    placeholders = job.get("placeholders") or {}
    audio_filename = placeholders.get("audio_filename") or f"audio_{int(time.time())}.mp3"
    json_placeholder = placeholders.get("json_placeholder") or f"tts_{int(time.time())}.json"

    selected_voice = job.get("selected_voice") or {}
    favorite_slug = selected_voice.get("favorite_slug") or selected_voice.get("favoriteSlug")
    voice_id = selected_voice.get("voice_id") or selected_voice.get("voiceId")
    engine_id = selected_voice.get("engine")
    provider = (job.get("preferred_provider") or "local").lower()

    # Build summarizer and generate audio
    try:
        from youtube_summarizer import YouTubeSummarizer  # heavy, but provides chunking + JSON update
    except Exception as exc:
        logging.error(f"Failed to import YouTubeSummarizer: {exc}")
        move_job(path, "failed")
        return False

    try:
        summarizer = YouTubeSummarizer()
    except Exception as exc:
        logging.error(f"Failed to init summarizer: {exc}")
        move_job(path, "failed")
        return False

    logging.info(
        f"Processing job {path.name} | provider={provider} fav={favorite_slug} voice={voice_id} engine={engine_id}"
    )

    try:
        result_path = await summarizer.generate_tts_audio(
            text,
            audio_filename,
            json_placeholder,
            provider=provider,
            voice=voice_id,
            engine=engine_id,
            favorite_slug=favorite_slug,
        )
    except Exception as exc:
        logging.error(f"TTS failed for {path.name}: {exc}")
        move_job(path, "failed")
        return False

    if not result_path or not Path(result_path).exists():
        logging.error(f"No audio produced for {path.name}")
        move_job(path, "failed")
        return False

    logging.info(f"Audio ready: {result_path}")

    # For summary jobs, perform DB/Render sync like the bot does
    if mode == "summary_audio":
        try:
            from nas_sync import dual_sync_upload  # type: ignore
            from modules.render_api_client import create_client_from_env  # type: ignore
        except Exception as exc:
            logging.warning(f"Sync helpers unavailable: {exc}")
            dual_sync_upload = None
            create_client_from_env = None

        video_info = job.get("video_info") or {}
        video_id = video_info.get("video_id") or video_info.get("id") or ""
        report_path = find_report_path(video_id)
        if dual_sync_upload and report_path:
            try:
                dual_sync_upload(report_path, Path(result_path))
                logging.info(f"Synced report+audio for {video_id}")
            except Exception as exc:
                logging.warning(f"Dual-sync failed for {video_id}: {exc}")
        else:
            logging.warning("Skipping dual-sync (no report found or helper unavailable)")

        if create_client_from_env and video_id:
            try:
                client = create_client_from_env()
                content_id = f"yt:{video_id}"
                client.upload_audio_file(Path(result_path), content_id)
                logging.info(f"Uploaded audio to Render for {content_id}")
            except Exception as exc:
                logging.warning(f"Render upload failed: {exc}")

    move_job(path, "processed")
    return True


def drain_once() -> int:
    jobs = sorted(QUEUE_DIR.glob("*.json"))
    if not jobs:
        logging.info("No queued jobs found.")
        return 0
    ok = 0
    for job_path in jobs:
        try:
            res = asyncio.run(process_job(job_path))
            ok += int(bool(res))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.error(f"Unhandled error processing {job_path.name}: {exc}")
            move_job(job_path, "failed")
    return ok


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Drain queued TTS jobs")
    parser.add_argument("--watch", action="store_true", help="Keep watching and draining the queue")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval seconds when --watch is set")
    args = parser.parse_args()

    if not QUEUE_DIR.exists():
        logging.info(f"Creating queue dir: {QUEUE_DIR}")
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.watch:
        count = drain_once()
        logging.info(f"Processed {count} job(s)")
        return 0

    logging.info(f"Watching queue at {QUEUE_DIR} (interval={args.interval}s)")
    try:
        while True:
            drain_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())


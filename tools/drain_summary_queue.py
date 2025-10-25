#!/usr/bin/env python3
"""Drain queued summary jobs once or in a loop.

Reads JSON jobs from data/summary_queue/, generates summaries via the Ollama
provider, saves reports, updates the ledger, and runs dual-sync where possible.

Usage:
  python tools/drain_summary_queue.py            # process queued jobs once
  python tools/drain_summary_queue.py --watch    # poll and process continuously
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is on sys.path so `modules` imports resolve when executed from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.summary_queue import QUEUE_DIR  # noqa: E402
from modules.services.summary_service import _is_local_summary_unavailable  # noqa: E402
from modules.services import sync_service  # noqa: E402
from modules import ledger  # noqa: E402
from modules.report_generator import JSONReportGenerator, create_report_from_youtube_summarizer  # noqa: E402
from youtube_summarizer import YouTubeSummarizer  # noqa: E402


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - drain_summary_queue - %(levelname)s - %(message)s",
    )


def load_job(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.error(f"Failed to read job {path.name}: {exc}")
        return None


def move_job(path: Path, outcome: str) -> None:
    dest_dir = QUEUE_DIR / outcome
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            logging.warning(f"Job {path.name} already moved; skipping move to {outcome}")
            return
        path.rename(dest_dir / path.name)
    except Exception as exc:
        logging.warning(f"Could not move job {path.name} to {outcome}: {exc}")


async def process_job(path: Path) -> bool:
    job = load_job(path)
    if not job:
        move_job(path, "failed")
        return False

    source = (job.get("source") or "youtube").lower()
    url = job.get("url")
    summary_type = job.get("summary_type") or "comprehensive"
    proficiency_level = job.get("proficiency_level")
    provider = (job.get("provider") or "ollama").lower()
    model = job.get("model")
    content_id = job.get("content_id")

    if not url:
        logging.error(f"Job {path.name} missing URL; marking as failed")
        move_job(path, "failed")
        return False

    if provider != "ollama":
        logging.error(f"Unsupported provider {provider} in {path.name}; marking as failed")
        move_job(path, "failed")
        return False

    try:
        summarizer = YouTubeSummarizer(llm_provider="ollama", model=model)
    except Exception as exc:
        logging.warning(f"Could not initialize Ollama summarizer: {exc}")
        return False

    logging.info(
        "▶️ Processing summary job %s | source=%s | summary=%s | model=%s",
        path.name,
        source,
        summary_type,
        getattr(summarizer, "model", None),
    )

    try:
        if source == "reddit":
            result = await summarizer.process_reddit_thread(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level,
            )
        elif source == "web":
            result = await summarizer.process_web_page(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level,
            )
        else:
            result = await summarizer.process_video(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level,
            )
    except Exception as exc:
        if _is_local_summary_unavailable(exc):
            logging.warning("Ollama summarizer unavailable: %s — deferring job", exc)
            return False
        logging.error(f"Summary generation failed for {path.name}: {exc}")
        move_job(path, "failed")
        return False

    if not result or (isinstance(result, dict) and result.get("error")):
        logging.error(f"Summarizer returned error for {path.name}: {result.get('error') if isinstance(result, dict) else 'unknown'}")
        move_job(path, "failed")
        return False

    generator = JSONReportGenerator("./data/reports")
    try:
        report_dict = create_report_from_youtube_summarizer(result)
        json_path = Path(generator.save_report(report_dict))
    except Exception as exc:
        logging.error(f"Failed to export report for {path.name}: {exc}")
        move_job(path, "failed")
        return False

    if not json_path.exists():
        logging.error(f"Exported report missing for {path.name}: {json_path}")
        move_job(path, "failed")
        return False

    stem = json_path.stem
    ledger_id = content_id or result.get("id") or stem
    now_iso = datetime.utcnow().isoformat()
    ledger_entry = {
        "stem": stem,
        "json": str(json_path),
        "mp3": None,
        "synced": False,
        "created_at": now_iso,
    }
    if proficiency_level:
        ledger_entry["proficiency"] = proficiency_level

    ledger.upsert(ledger_id, summary_type, ledger_entry)
    logging.info("📄 Saved report %s and updated ledger for %s:%s", json_path.name, ledger_id, summary_type)

    outcome = sync_service.run_dual_sync(json_path, label=result.get("id"))
    if outcome["success"]:
        sync_service.update_ledger_after_sync(
            ledger_id,
            summary_type,
            targets=outcome["targets"],
        )
    else:
        logging.warning("Dual-sync failed for %s: %s", ledger_id, outcome.get("error"))

    move_job(path, "processed")
    return True


def drain_once() -> int:
    jobs = sorted(QUEUE_DIR.glob("*.json"))
    if not jobs:
        logging.info("No queued summary jobs found.")
        return 0
    ok = 0
    for job_path in jobs:
        try:
            if asyncio.run(process_job(job_path)):
                ok += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.error(f"Unhandled error processing {job_path.name}: {exc}")
            move_job(job_path, "failed")
    return ok


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Drain queued summary jobs")
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

    logging.info(f"Watching summary queue at {QUEUE_DIR} (interval={args.interval}s)")
    try:
        while True:
            drain_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())

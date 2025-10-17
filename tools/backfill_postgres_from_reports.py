#!/usr/bin/env python3
"""
Backfill Postgres directly from existing JSON reports.

This script walks the reports directory, loads each JSON file, normalizes any
legacy summary structure into the modern `summary.variants` list, and then
uses `PostgresWriter` to upsert the content + summary variants. Optionally it
can attach audio variants if MP3 files are present and `AUDIO_PUBLIC_BASE` is
configured.

Usage (run inside the container so psycopg is available):
    python tools/backfill_postgres_from_reports.py --resume

Flags:
    --reports-dir PATH   Directory of JSON reports (default: data/reports)
    --limit N            Process at most N files (helpful for testing)
    --resume             Resume using a state file (.postgres_backfill_state.json)
    --audio              Also attempt to insert audio variants (needs AUDIO_PUBLIC_BASE)
    --dry-run            Parse & normalize only; do not write to Postgres
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from modules.postgres_writer import PostgresWriter
from modules.summary_variants import (
    format_summary_html,
    normalize_variant_id,
    variant_kind,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BackfillRunner:
    def __init__(
        self,
        reports_dir: Path,
        resume: bool = False,
        dry_run: bool = False,
        attach_audio: bool = False,
    ):
        self.reports_dir = reports_dir
        self.resume = resume
        self.dry_run = dry_run
        self.attach_audio = attach_audio
        self.state_file = self.reports_dir / ".postgres_backfill_state.json"
        self.processed: Set[str] = set()
        if resume:
            self._load_state()

        self.writer = PostgresWriter()

        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "uploaded": 0,
            "errors": 0,
            "audio_uploaded": 0,
        }

    # --- state handling -------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            prev = data.get("processed_files", [])
            if isinstance(prev, list):
                self.processed.update(p for p in prev if isinstance(p, str))
            logger.info("Resume enabled – skipping %d files already processed", len(self.processed))
        except Exception as exc:
            logger.warning("Could not read state file %s: %s", self.state_file, exc)

    def _save_state(self) -> None:
        if not self.resume:
            return
        payload = {
            "processed_files": sorted(self.processed),
            "last_saved": datetime.utcnow().isoformat(),
            "stats": self.stats,
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.state_file)

    # --- core workflow --------------------------------------------------
    def run(self, limit: Optional[int] = None) -> bool:
        if not self.reports_dir.is_dir():
            logger.error("Reports directory not found: %s", self.reports_dir)
            return False

        files = sorted(
            f for f in self.reports_dir.glob("*.json")
            if not f.name.startswith("._")
        )
        self.stats["total"] = len(files)
        logger.info("Found %d JSON reports", len(files))

        for idx, path in enumerate(files, start=1):
            if limit and self.stats["processed"] >= limit:
                logger.info("Reached limit (%d files). Stopping.", limit)
                break

            if path.name in self.processed:
                self.stats["skipped"] += 1
                continue

            logger.info("[%d/%d] Processing %s", idx, len(files), path.name)
            self.stats["processed"] += 1

            try:
                report = self._load_report(path)
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path.name, exc)
                self.stats["errors"] += 1
                continue

            try:
                normalized = self._normalize_report(report)
            except Exception as exc:
                logger.error("Failed to normalize %s: %s", path.name, exc)
                self.stats["errors"] += 1
                continue

            if self.dry_run:
                logger.info("Dry run – skipping database write for %s", path.name)
                self.processed.add(path.name)
                self._save_state()
                continue

            try:
                result = self.writer.upload_content(normalized)
                if result and result.get("upserted"):
                    self.stats["uploaded"] += 1
                    if self.attach_audio:
                        vid = (
                            result.get("video_id")
                            or normalized.get("video_id")
                            or (normalized.get("id", "").split(":", 1)[-1] if normalized.get("id") else None)
                        )
                        if not vid:
                            logger.warning("No video_id found for audio upload on %s", path.name)
                        else:
                            audio_result = self.writer.upload_audio(vid)
                            if audio_result:
                                self.stats["audio_uploaded"] += 1
                else:
                    logger.warning("Upload returned no result for %s", normalized.get("video_id"))
            except Exception as exc:
                logger.error("Database write failed for %s: %s", path.name, exc)
                self.stats["errors"] += 1
                continue

            self.processed.add(path.name)
            self._save_state()

        logger.info("Backfill complete. Uploaded %d files (%d errors, %d skipped).",
                    self.stats["uploaded"], self.stats["errors"], self.stats["skipped"])
        if self.attach_audio and self.stats["audio_uploaded"]:
            logger.info("Audio variants inserted for %d videos.", self.stats["audio_uploaded"])
        return self.stats["errors"] == 0

    # --- helpers --------------------------------------------------------
    def _load_report(self, path: Path) -> Dict[str, Any]:
        encodings = ("utf-8", "utf-8-sig", "latin-1")
        last_exc: Optional[Exception] = None
        for encoding in encodings:
            try:
                return json.loads(path.read_text(encoding=encoding))
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"could not decode JSON ({last_exc})")

    def _normalize_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Return report dict ready for PostgresWriter."""
        # Ensure summary structure is a dict
        summary_section = report.get("summary")
        if summary_section is None or not isinstance(summary_section, dict):
            summary_section = {}
            report["summary"] = summary_section

        # If variants already exist with usable entries, keep them
        variants = self._build_variants(report, summary_section)
        if variants:
            summary_section["variants"] = variants
            report["summary_variants"] = variants

        # Ensure summary.summary exists for fallback eligibility
        if not summary_section.get("summary") and variants:
            summary_section["summary"] = variants[0]["text"]
            summary_section["summary_type"] = variants[0]["variant"]

        return report

    def _build_variants(
        self,
        report: Dict[str, Any],
        summary_section: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Convert legacy summary fields into normalized variant entries."""
        variants: List[Dict[str, Any]] = []
        processed: Set[str] = set()

        def add_variant(variant_id: str, text: str, base: Dict[str, Any]) -> None:
            norm_id = normalize_variant_id(variant_id)
            if not norm_id or norm_id in processed:
                return
            clean_text = (text or "").strip()
            if not clean_text:
                return
            entry = {
                "variant": norm_id,
                "text": clean_text,
                "html": base.get("html") or format_summary_html(clean_text),
                "kind": base.get("kind") or variant_kind(norm_id),
            }
            for key in ("headline", "language", "generated_at", "proficiency", "proficiency_level", "audio_url"):
                if base.get(key):
                    entry[key if key != "proficiency_level" else "proficiency"] = base[key]
            variants.append(entry)
            processed.add(norm_id)

        # 1) Existing variants array
        existing_variants = summary_section.get("variants")
        if isinstance(existing_variants, list):
            for entry in existing_variants:
                if not isinstance(entry, dict):
                    continue
                add_variant(
                    entry.get("variant") or entry.get("summary_type") or entry.get("type"),
                    entry.get("text") or entry.get("summary") or entry.get("content") or "",
                    entry,
                )

        # 2) Top-level summary_variants (some reports store it there)
        top_variants = report.get("summary_variants")
        if isinstance(top_variants, list):
            for entry in top_variants:
                if not isinstance(entry, dict):
                    continue
                add_variant(
                    entry.get("variant"),
                    entry.get("text") or entry.get("summary") or entry.get("content") or "",
                    entry,
                )

        # 3) Direct summary text
        direct_text = summary_section.get("summary") or report.get("summary_text")
        direct_variant = summary_section.get("summary_type") or summary_section.get("default_variant")
        add_variant(direct_variant or "comprehensive", direct_text or "", summary_section)

        # 4) Legacy named keys (bullet_points, key_insights, etc.)
        legacy_ignore = {
            "summary",
            "headline",
            "summary_type",
            "generated_at",
            "variants",
            "default_variant",
            "latest_variant",
            "language",
            "proficiency",
            "proficiency_level",
            "audio_url",
            "html",
            "text",
        }
        for key, value in summary_section.items():
            if key in legacy_ignore:
                continue
            if not isinstance(value, str):
                continue
            add_variant(key, value, summary_section)

        return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Postgres from JSON reports.")
    parser.add_argument("--reports-dir", default="data/reports", help="Directory containing report JSON files.")
    parser.add_argument("--limit", type=int, help="Process at most N files.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state file.")
    parser.add_argument("--audio", action="store_true", help="Attach audio variants when files are present.")
    parser.add_argument("--dry-run", action="store_true", help="Parse/normalize only; do not write to Postgres.")
    args = parser.parse_args()

    runner = BackfillRunner(
        reports_dir=Path(args.reports_dir),
        resume=args.resume,
        dry_run=args.dry_run,
        attach_audio=args.audio,
    )

    success = runner.run(limit=args.limit)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

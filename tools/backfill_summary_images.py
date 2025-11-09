#!/usr/bin/env python3
"""
Generate summary images for existing records that are missing one.

Usage:
    python tools/backfill_summary_images.py --limit 25

Environment:
    DATABASE_URL_POSTGRES_NEW (preferred) or DATABASE_URL for Postgres DSN
    RENDER_DASHBOARD_URL / INGEST_TOKEN / SYNC_SECRET for Render uploads
    TTSHUB_API_BASE pointing at the Draw Things hub

The script walks the `content` table, finds rows with an empty
`summary_image_url`, generates an illustration via the existing
summary_image_service, uploads the PNG to the dashboard, and updates the
database field. Runs sequentially to keep hub load low.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from psycopg.rows import dict_row

from modules.render_api_client import RenderAPIClient
from modules.services import summary_image_service

# Some environments leave the flag off; enable it by default for backfill runs.
os.environ.setdefault("SUMMARY_IMAGE_ENABLED", "1")

LOGGER = logging.getLogger("backfill_summary_images")

LOGGER.info("backfill_summary_images.py loaded from repo commit")

IMAGE_MODES = ("ai1", "ai2")

SUMMARY_SQL = """
WITH ranked AS (
    SELECT
        vs.video_id,
        vs.variant,
        vs.text,
        ROW_NUMBER() OVER (
            PARTITION BY vs.video_id
            ORDER BY
                CASE
                    WHEN vs.variant = 'comprehensive' THEN 1
                    WHEN vs.variant = 'key-insights' THEN 2
                    WHEN vs.variant = 'bullet-points' THEN 3
                    WHEN vs.variant = 'audio' THEN 4
                    ELSE 5
                END,
                vs.created_at DESC
        ) AS rn
    FROM v_latest_summaries vs
)
SELECT
    c.id,
    c.video_id,
    c.title,
    c.analysis_json,
    c.subcategories_json,
    r.text AS summary_text,
    c.summary_image_url,
    c.thumbnail_url,
    c.indexed_at
FROM content c
LEFT JOIN ranked r
    ON r.video_id = c.video_id AND r.rn = 1
WHERE
    {image_clause}
    {thumbnail_filter}
ORDER BY c.indexed_at DESC
LIMIT %(limit)s;
"""

SUMMARY_SQL_ANY = """
WITH ranked AS (
    SELECT
        vs.video_id,
        vs.variant,
        vs.text,
        ROW_NUMBER() OVER (
            PARTITION BY vs.video_id
            ORDER BY
                CASE
                    WHEN vs.variant = 'comprehensive' THEN 1
                    WHEN vs.variant = 'key-insights' THEN 2
                    WHEN vs.variant = 'bullet-points' THEN 3
                    WHEN vs.variant = 'audio' THEN 4
                    ELSE 5
                END,
                vs.created_at DESC
        ) AS rn
    FROM v_latest_summaries vs
)
SELECT
    c.id,
    c.video_id,
    c.title,
    c.analysis_json,
    c.subcategories_json,
    r.text AS summary_text,
    c.summary_image_url,
    c.thumbnail_url,
    c.indexed_at
FROM content c
LEFT JOIN ranked r
    ON r.video_id = c.video_id AND r.rn = 1
WHERE
    {image_clause}
    {thumbnail_filter}
ORDER BY c.indexed_at DESC
LIMIT %(limit)s;
"""


@dataclass
class TaskItem:
    video_id: str
    title: str
    summary_text: str
    analysis: Dict[str, Any]
    indexed_at: Any


def _has_ai2_variant(analysis: Dict[str, Any]) -> bool:
    variants = analysis.get("summary_image_variants")
    if not isinstance(variants, list):
        return False
    for entry in variants:
        if not isinstance(entry, dict):
            continue
        if entry.get("image_mode") == "ai2":
            return True
        template = (entry.get("template") or "").lower()
        if template == "ai2_freestyle":
            return True
        url = (entry.get("url") or entry.get("relative_path") or "").lower()
        if url.startswith("ai2_"):
            return True
    return False


def _fetch_candidates(
    conn: psycopg.Connection,
    limit: int,
    only_missing_thumbnail: bool,
    video_ids: Optional[List[str]] = None,
    mode: str = "ai1",
) -> List[TaskItem]:
    mode = (mode or "ai1").lower()
    image_clause = "1=1" if mode == "ai2" else "(c.summary_image_url IS NULL OR c.summary_image_url = '')"
    if video_ids:
        placeholders = ", ".join(["%s"] * len(video_ids))
        sql = (
            """
            WITH ranked AS (
                SELECT
                    vs.video_id,
                    vs.variant,
                    vs.text,
                    ROW_NUMBER() OVER (
                        PARTITION BY vs.video_id
                        ORDER BY vs.created_at DESC
                    ) AS rn
                FROM v_latest_summaries vs
                WHERE vs.video_id IN ("""
            + placeholders
            + """)
            )
            SELECT
                c.id,
                c.video_id,
                c.title,
                c.analysis_json,
                c.subcategories_json,
                r.text AS summary_text,
                c.summary_image_url,
                c.thumbnail_url,
                c.indexed_at
            FROM content c
            LEFT JOIN ranked r
                ON r.video_id = c.video_id AND r.rn = 1
            WHERE c.video_id IN ("""
            + placeholders
            + """);
            """
        )
        params = tuple(video_ids) * 2
    else:
        thumbnail_clause = (
            "AND (c.thumbnail_url IS NULL OR c.thumbnail_url = '')" if only_missing_thumbnail else ""
        )
        sql_template = SUMMARY_SQL_ANY if mode == "ai2" else SUMMARY_SQL
        sql = sql_template.format(image_clause=image_clause, thumbnail_filter=thumbnail_clause)
        params = {"limit": limit}

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    tasks: List[TaskItem] = []
    for row in rows:
        summary_text = (row.get("summary_text") or "").strip()
        if not summary_text:
            LOGGER.debug(
                "Skipping %s – no summary text available",
                row.get("video_id"),
            )
            continue

        lowered = summary_text.lower()
        if lowered.startswith("unable to generate") or lowered.startswith("unable to create") or lowered.startswith("summary generation failed"):
            LOGGER.debug(
                "Skipping %s – summary text placeholder detected",
                row.get("video_id"),
            )
            continue

        analysis = row.get("analysis_json") or {}
        if isinstance(analysis, str):
            # psycopg may return json as str in some configs; attempt to eval
            try:
                import json

                analysis = json.loads(analysis)
            except Exception:
                analysis = {}
        skip_ai2 = mode == "ai2" and not video_ids and _has_ai2_variant(analysis if isinstance(analysis, dict) else {})
        if skip_ai2:
            LOGGER.debug("Skipping %s – AI2 variant already present", row.get("video_id"))
            continue

        tasks.append(
            TaskItem(
                video_id=row["video_id"],
                title=row.get("title") or "",
                summary_text=summary_text,
                analysis=analysis if isinstance(analysis, dict) else {},
                indexed_at=row.get("indexed_at"),
            )
        )

    return tasks


async def _generate_image(
    task: TaskItem,
    *,
    mode: str,
) -> Optional[Dict[str, Any]]:
    payload = {
        "id": task.video_id,
        "title": task.title,
        "summary": {"summary": task.summary_text},
        "analysis": task.analysis,
    }
    try:
        metadata = await summary_image_service.maybe_generate_summary_image(payload, mode=mode)
        return metadata
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Image generation failed for %s: %s", task.video_id, exc)
        return None


def _update_summary_image_metadata(
    conn: psycopg.Connection,
    video_id: str,
    url: str,
    metadata: Dict[str, Any],
    *,
    mode: str,
) -> None:
    analysis_blob = None
    existing_url: Optional[str] = None
    with conn.cursor() as cur:
        cur.execute("SELECT analysis_json, summary_image_url FROM content WHERE video_id=%s", (video_id,))
        row = cur.fetchone()
        if row:
            analysis_blob = row[0]
            existing_url = row[1]
    analysis = {}
    if analysis_blob:
        if isinstance(analysis_blob, str):
            try:
                analysis = json.loads(analysis_blob)
            except Exception:
                analysis = {}
        elif isinstance(analysis_blob, dict):
            analysis = analysis_blob
    variant_entry = metadata.get("analysis_variant")
    if not variant_entry:
        variant_entry = summary_image_service.build_analysis_variant(metadata, url)
    else:
        if not variant_entry.get("url"):
            variant_entry = dict(variant_entry)
            variant_entry["url"] = url
        if mode == "ai2":
            variant_entry = dict(variant_entry)
            variant_entry["image_mode"] = "ai2"
    selected_url = url if mode == "ai1" else None
    updated_analysis = summary_image_service.apply_analysis_variant(
        analysis,
        variant_entry,
        selected_url=selected_url,
        prompt=metadata.get("prompt"),
        model=metadata.get("model"),
    )
    payload = json.dumps(updated_analysis)
    new_summary_url = url if mode == "ai1" else (existing_url or "")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE content
               SET summary_image_url = %s,
                   analysis_json = %s,
                   updated_at = now()
             WHERE video_id = %s;
            """,
            (new_summary_url, payload, video_id),
        )
    conn.commit()


async def process_tasks(
    tasks: List[TaskItem],
    render_client: RenderAPIClient,
    conn: psycopg.Connection,
    *,
    dry_run: bool,
    delay_seconds: float,
    mode: str,
) -> None:
    total = len(tasks)
    for idx, task in enumerate(tasks, 1):
        LOGGER.info("(%d/%d) Processing %s", idx, total, task.video_id)

        metadata = await _generate_image(task, mode=mode)
        if not metadata:
            LOGGER.warning("No image generated for %s", task.video_id)
            continue

        prompt_snippet = (metadata.get("prompt") or "").strip()
        if prompt_snippet:
            prompt_snippet = prompt_snippet.replace("\n", " ").strip()
        if prompt_snippet:
            LOGGER.info(
                "Prompt (%s/%s): %s%s",
                metadata.get("template") or "default",
                metadata.get("prompt_source") or metadata.get("template") or "template",
                prompt_snippet[:240],
                "…" if len(prompt_snippet) > 240 else "",
            )

        image_path = Path(metadata["path"])
        if not image_path.exists():
            LOGGER.error("Generated image path missing for %s: %s", task.video_id, image_path)
            continue

        if dry_run:
            LOGGER.info(
                "Dry run – would upload %s (template=%s, mode=%s)",
                image_path,
                metadata.get("template"),
                mode,
            )
            try:
                image_path.unlink()
            except Exception as exc:
                LOGGER.debug("Unable to remove temporary image %s: %s", image_path, exc)
        else:
            try:
                upload_info = render_client.upload_image_file(image_path, task.video_id)
            except Exception as exc:
                LOGGER.error("Render upload failed for %s: %s", task.video_id, exc)
                continue

            summary_image_url = (
                upload_info.get("public_url")
                or upload_info.get("relative_url")
                or metadata.get("public_url")
                or metadata.get("relative_path")
            )

            if not summary_image_url:
                LOGGER.error("No URL returned for %s upload", task.video_id)
                continue

            try:
                _update_summary_image_metadata(conn, task.video_id, summary_image_url, metadata or {}, mode=mode)
            except Exception as exc:
                LOGGER.error("Failed to update DB for %s: %s", task.video_id, exc)
                continue

            LOGGER.info("✅ Image synced for %s → %s", task.video_id, summary_image_url)

        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill summary images for existing content.")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of rows to process (default: 25).",
    )
    parser.add_argument(
        "--only-missing-thumbnail",
        action="store_true",
        help="Restrict to rows where thumbnail_url is null/empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate images but skip upload/DB updates.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between uploads (default: 0.2).",
    )
    parser.add_argument(
        "--mode",
        choices=IMAGE_MODES,
        default="ai1",
        help="Image style to generate (ai1=themed, ai2=freestyle).",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Target specific video_id (can be repeated). When provided, --limit is ignored and only these IDs are processed.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    args = parse_args()
    dsn = os.getenv("DATABASE_URL_POSTGRES_NEW") or os.getenv("DATABASE_URL")
    if not dsn:
        LOGGER.error("DATABASE_URL_POSTGRES_NEW or DATABASE_URL must be set.")
        return 1

    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:
        LOGGER.error("Failed to connect to Postgres: %s", exc)
        return 1

    try:
        video_ids = args.video_ids
        mode = args.mode
        if video_ids:
            LOGGER.info("Targeted run for %d video_id(s) (mode=%s)", len(video_ids), mode)
        tasks = _fetch_candidates(conn, args.limit, args.only_missing_thumbnail, video_ids, mode)
    except Exception as exc:
        LOGGER.error("Failed to fetch candidates: %s", exc)
        conn.close()
        return 1

    if not tasks:
        LOGGER.info("No candidate rows found.")
        conn.close()
        return 0

    LOGGER.info("Loaded %d candidate(s).", len(tasks))

    try:
        render_client = RenderAPIClient()
    except Exception as exc:
        LOGGER.error("Failed to initialise Render client: %s", exc)
        conn.close()
        return 1

    try:
        asyncio.run(
            process_tasks(
                tasks,
                render_client,
                conn,
                dry_run=args.dry_run,
                delay_seconds=args.delay,
                mode=mode,
            )
        )
    finally:
        conn.close()

    LOGGER.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
PostgreSQL Writer for YTV2 NAS (Postgres-only dashboard)

This module writes directly to Postgres using UPSERTs, replacing the
deprecated HTTP ingest client. It exposes a drop-in compatible API for
DualSyncCoordinator: `health_check()`, `upload_content()`, and
`upload_audio()`.

Environment:
- DATABASE_URL (preferred) or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE/PGSSLMODE
- AUDIO_PUBLIC_BASE (optional): base URL used to build public audio links
  when inserting `audio` variants (e.g. https://your.host/exports)
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore

from .summary_variants import normalize_variant_id, variant_kind, format_summary_html

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _env_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    # Assemble from PG* vars if provided
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or "postgres"
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or ""
    dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "postgres"
    sslmode = os.getenv("PGSSLMODE", "require")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{dbname}?sslmode={sslmode}"


@dataclass
class DBColumns:
    has_id: bool
    has_language: bool
    has_topics_json: bool
    has_subcategories_json: bool
    has_analysis_json: bool


class PostgresWriter:
    """Direct Postgres writer with minimal, safe UPSERTs."""

    def __init__(self, dsn: Optional[str] = None):
        # Lazy-import psycopg in case it was installed after process start
        global psycopg
        if psycopg is None:
            try:
                import importlib
                psycopg = importlib.import_module('psycopg')
            except Exception as e:
                raise RuntimeError(
                    "psycopg is not installed. Please add 'psycopg[binary]>=3.1' to requirements.txt and rebuild."
                ) from e
        self.dsn = dsn or _env_dsn()
        self._columns_cache: Optional[DBColumns] = None
        self.audio_public_base = os.getenv("AUDIO_PUBLIC_BASE") or os.getenv("POSTGRES_DASHBOARD_URL")

    # --- Connection helpers ---
    def _connect(self):  # returns connection with autocommit
        return psycopg.connect(self.dsn, autocommit=True)

    def health_check(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception as e:
            logger.error(f"Postgres health_check failed: {e}")
            return False

    # --- Schema discovery ---
    def _discover_columns(self) -> DBColumns:
        if self._columns_cache:
            return self._columns_cache
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    select column_name from information_schema.columns
                    where table_name = 'content'
                    """
                )
                cols = {row[0] for row in cur.fetchall()}
        except Exception as e:
            logger.warning(f"Could not introspect 'content' columns, assuming minimal set. Error: {e}")
            cols = set()
        self._columns_cache = DBColumns(
            has_id=("id" in cols) if cols else True,
            has_language=("language" in cols) if cols else True,
            has_topics_json=("topics_json" in cols) if cols else True,
            has_subcategories_json=("subcategories_json" in cols) if cols else True,
            has_analysis_json=("analysis_json" in cols) if cols else True,
        )
        return self._columns_cache

    # --- Payload conversion ---
    def _to_db_payload(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten universal schema into columns + summary variants list.

        Returns keys: video_id, id (optional), title, channel_name, canonical_url,
        thumbnail_url, duration_seconds, indexed_at (datetime), has_audio, language,
        analysis_json, subcategories_json, topics_json, summary_variants (list).
        """
        # video_id
        video_id = (
            content_data.get("video_id")
            or (content_data.get("id", "").split(":", 1)[-1] if content_data.get("id") else None)
            or content_data.get("source_metadata", {}).get("youtube", {}).get("video_id")
            or content_data.get("source_metadata", {}).get("reddit", {}).get("id")
        )
        if not video_id:
            raise ValueError("video_id is required to upsert content")
        raw_id = content_data.get("id") or f"yt:{video_id}"

        # simple fields with fallbacks
        title = (
            content_data.get("title")
            or content_data.get("source_metadata", {}).get("youtube", {}).get("title")
            or content_data.get("summary", {}).get("headline")
            or f"Content {video_id}"
        )
        channel_name = (
            content_data.get("channel_name")
            or content_data.get("source_metadata", {}).get("youtube", {}).get("channel_name")
            or content_data.get("uploader")
        )
        canonical_url = content_data.get("canonical_url") or content_data.get("source_metadata", {}).get("youtube", {}).get("canonical_url")
        if not canonical_url and video_id:
            canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail_url = (
            content_data.get("thumbnail_url")
            or content_data.get("source_metadata", {}).get("youtube", {}).get("thumbnail_url")
        )
        duration_seconds = (
            content_data.get("duration_seconds")
            or content_data.get("media_info", {}).get("audio_duration_seconds")
            or content_data.get("source_metadata", {}).get("youtube", {}).get("duration_seconds")
            or 0
        )

        # time & booleans
        indexed_at_raw = (
            content_data.get("indexed_at")
            or content_data.get("processing", {}).get("completed_at")
            or content_data.get("processing_metadata", {}).get("indexed_at")
        )
        try:
            indexed_at = datetime.fromisoformat(str(indexed_at_raw).replace("Z", "+00:00")) if indexed_at_raw else _now_utc()
        except Exception:
            indexed_at = _now_utc()
        has_audio = bool(
            content_data.get("has_audio")
            or content_data.get("media_info", {}).get("has_audio")
            or any(
                (variant_kind(normalize_variant_id(v.get("variant"))) == "audio")
                for v in (content_data.get("summary", {}).get("variants") or [])
                if isinstance(v, dict)
            )
        )

        # language
        language = (
            content_data.get("language")
            or content_data.get("content_analysis", {}).get("language")
            or content_data.get("summary", {}).get("language")
            or content_data.get("analysis", {}).get("language")
        )

        # JSONBs
        analysis_json = content_data.get("analysis_json") or content_data.get("analysis") or content_data.get("content_analysis")
        subcats = content_data.get("subcategories_json") or (
            (content_data.get("analysis", {}) or content_data.get("content_analysis", {})).get("categories")
            if isinstance(content_data.get("analysis"), dict) or isinstance(content_data.get("content_analysis"), dict)
            else None
        )
        if subcats and not isinstance(subcats, dict):
            subcats = {"categories": subcats}
        topics_json = content_data.get("topics_json")

        # Variants
        summary_variants: List[Dict[str, Any]] = []
        summary_section = content_data.get("summary") or {}
        default_variant = normalize_variant_id(summary_section.get("default_variant") or summary_section.get("summary_type") or "comprehensive")
        variants_field = summary_section.get("variants")
        if isinstance(variants_field, list):
            for entry in variants_field:
                if not isinstance(entry, dict):
                    continue
                variant_id = normalize_variant_id(entry.get("variant") or entry.get("summary_type") or entry.get("type"))
                if not variant_id:
                    continue
                text_value = entry.get("text") or entry.get("summary") or entry.get("content")
                if not isinstance(text_value, str) or not text_value.strip():
                    continue
                variant_payload = {
                    "variant": variant_id,
                    "text": text_value.strip(),
                    "html": entry.get("html") or format_summary_html(text_value),
                    "language": entry.get("language") or summary_section.get("language") or language,
                    "headline": entry.get("headline") or summary_section.get("headline"),
                    "proficiency": entry.get("proficiency") or entry.get("proficiency_level"),
                }
                if entry.get("generated_at"):
                    variant_payload["generated_at"] = entry["generated_at"]
                if entry.get("audio_url"):
                    variant_payload["audio_url"] = entry["audio_url"]
                summary_variants.append(variant_payload)

        # Promote single summary to variant if none provided
        if not summary_variants:
            direct_text = summary_section.get("summary") or content_data.get("summary_text")
            if isinstance(direct_text, str) and direct_text.strip():
                summary_variants.append({
                    "variant": default_variant,
                    "text": direct_text.strip(),
                    "html": format_summary_html(direct_text),
                    "language": language,
                })

        return {
            "video_id": video_id,
            "id": raw_id,
            "title": title,
            "channel_name": channel_name,
            "canonical_url": canonical_url,
            "thumbnail_url": thumbnail_url,
            "duration_seconds": int(duration_seconds) if duration_seconds else None,
            "indexed_at": indexed_at,
            "has_audio": has_audio,
            "language": language,
            "analysis_json": analysis_json,
            "subcategories_json": subcats,
            "topics_json": topics_json,
            "summary_variants": summary_variants,
        }

    # --- Public, coordinator-compatible API ---
    def upload_content(self, content_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Upsert into content and write summary variants.

        Returns a small result dict similar to HTTP client: {'upserted': True, 'video_id': ...}
        """
        payload = self._to_db_payload(content_data)
        cols = self._discover_columns()

        # Build dynamic insert list depending on existing columns
        insert_cols = [
            "video_id",
            "title",
            "channel_name",
            "canonical_url",
            "thumbnail_url",
            "duration_seconds",
            "indexed_at",
            "has_audio",
        ]
        if cols.has_id:
            insert_cols.insert(0, "id")
        if cols.has_language:
            insert_cols.append("language")
        if cols.has_analysis_json:
            insert_cols.append("analysis_json")
        if cols.has_subcategories_json:
            insert_cols.append("subcategories_json")
        if cols.has_topics_json:
            insert_cols.append("topics_json")

        # Prepare parameter dict with JSON serialization where needed
        params: Dict[str, Any] = {k: payload.get(k) for k in insert_cols}
        if params.get("analysis_json") is not None and not isinstance(params["analysis_json"], (str, bytes)):
            params["analysis_json"] = json.dumps(params["analysis_json"])  # psycopg handles json, but keep simple
        if params.get("subcategories_json") is not None and not isinstance(params["subcategories_json"], (str, bytes)):
            params["subcategories_json"] = json.dumps(params["subcategories_json"]) 
        if params.get("topics_json") is not None and not isinstance(params["topics_json"], (str, bytes)):
            params["topics_json"] = json.dumps(params["topics_json"]) 

        # Build SQL strings
        cols_sql = ", ".join(insert_cols)
        vals_sql = ", ".join([f"%({c})s" for c in insert_cols])
        set_updates = [
            "title = EXCLUDED.title",
            "channel_name = EXCLUDED.channel_name",
            "canonical_url = EXCLUDED.canonical_url",
            "thumbnail_url = EXCLUDED.thumbnail_url",
            "duration_seconds = EXCLUDED.duration_seconds",
            "indexed_at = EXCLUDED.indexed_at",
            "has_audio = EXCLUDED.has_audio",
        ]
        if cols.has_language:
            set_updates.append("language = EXCLUDED.language")
        if cols.has_analysis_json:
            set_updates.append("analysis_json = EXCLUDED.analysis_json")
        if cols.has_subcategories_json:
            set_updates.append("subcategories_json = EXCLUDED.subcategories_json")
        if cols.has_topics_json:
            set_updates.append("topics_json = EXCLUDED.topics_json")
        set_updates.append("updated_at = now()")
        upsert_sql = (
            f"INSERT INTO content ({cols_sql}) VALUES ({vals_sql}) "
            f"ON CONFLICT (video_id) DO UPDATE SET {', '.join(set_updates)}"
        )

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(upsert_sql, params)
                # Upsert summary variants
                self._upsert_variants(cur, payload["video_id"], payload.get("summary_variants") or [])
            return {"upserted": True, "video_id": payload["video_id"]}
        except Exception as e:
            logger.error(f"Content upsert failed for {payload.get('video_id')}: {e}")
            return None

    def upload_audio(self, video_id: str, audio_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Insert/refresh an `audio` variant pointing to a public URL.

        If `AUDIO_PUBLIC_BASE` (or POSTGRES_DASHBOARD_URL) is set, the URL is
        built as f"{base}/exports/{basename}". Otherwise, returns None and logs
        a warning.
        """
        if not audio_path:
            # Try common patterns similar to previous client
            clean_video_id = video_id.replace("yt:", "").strip()
            roots = [Path("./exports"), Path("./data/exports")]
            candidates: List[Path] = []
            for root in roots:
                if root.is_dir():
                    candidates.extend(list(root.glob(f"audio_{clean_video_id}_*.mp3")))
                    candidates.extend(list(root.glob(f"*{clean_video_id}*.mp3")))
            if not candidates:
                logger.warning(f"No audio files found for {video_id}")
                return None
            audio_path = max(candidates, key=lambda p: p.stat().st_mtime)

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if not self.audio_public_base:
            logger.warning("AUDIO_PUBLIC_BASE not set; cannot publish audio URL. Skipping audio variant insert.")
            return None

        base = self.audio_public_base.rstrip("/")
        public_url = f"{base}/exports/{audio_path.name}"
        audio_html = f"<audio controls src=\"{public_url}\"></audio>"

        try:
            with self._connect() as conn, conn.cursor() as cur:
                # Insert/refresh audio summary variant (revision 1 by default)
                cur.execute(
                    """
                    INSERT INTO summaries (video_id, variant, revision, text, html, created_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (video_id, variant, revision) DO UPDATE SET
                      text = EXCLUDED.text,
                      html = EXCLUDED.html,
                      created_at = EXCLUDED.created_at
                    """,
                    (video_id, "audio", 1, None, audio_html),
                )
                # Flip content.has_audio
                cur.execute(
                    "UPDATE content SET has_audio = TRUE, updated_at = now() WHERE video_id = %s",
                    (video_id,),
                )
            return {"public_url": public_url, "upserted": True}
        except Exception as e:
            logger.error(f"Audio variant upsert failed for {video_id}: {e}")
            return None

    # --- internals ---
    def _upsert_variants(self, cur, video_id: str, variants: List[Dict[str, Any]]):
        if not variants:
            return
        for entry in variants:
            variant = normalize_variant_id(entry.get("variant"))
            if not variant:
                continue
            kind = variant_kind(variant)
            text_value = entry.get("text")
            html_value = entry.get("html") or (format_summary_html(text_value) if text_value else None)
            created_at = entry.get("generated_at")
            if created_at:
                try:
                    created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                except Exception:
                    created_at = None
            revision = int(entry.get("revision") or 1)

            # Ensure we always have some HTML for card eligibility when there is text
            if not html_value and isinstance(text_value, str) and text_value.strip():
                html_value = format_summary_html(text_value)

            cur.execute(
                """
                INSERT INTO summaries (video_id, variant, revision, text, html, created_at)
                VALUES (%s, %s, %s, %s, %s, COALESCE(%s, now()))
                ON CONFLICT (video_id, variant, revision) DO UPDATE SET
                  text = EXCLUDED.text,
                  html = EXCLUDED.html,
                  created_at = EXCLUDED.created_at
                """,
                (video_id, variant, revision, text_value, html_value, created_at),
            )


def create_postgres_writer_from_env() -> PostgresWriter:
    return PostgresWriter()


__all__ = ["PostgresWriter", "create_postgres_writer_from_env"]

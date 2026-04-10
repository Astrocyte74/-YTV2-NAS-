"""Persistence helpers for audio on-demand artifact generation."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore

logger = logging.getLogger(__name__)


def _dsn() -> str:
    return (
        os.getenv("DATABASE_URL")
        or f"postgresql://{os.getenv('PGUSER', 'ytv2')}:{os.getenv('PGPASSWORD', '')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}"
        f"/{os.getenv('PGDATABASE', 'ytv2')}"
    )


class AudioStore:
    """CRUD for audio_artifacts table."""

    def _connect(self):
        if psycopg is None:
            raise RuntimeError("psycopg not installed")
        from psycopg.rows import dict_row
        return psycopg.connect(_dsn(), row_factory=dict_row)

    # ---- Read ----

    def get_artifact(self, video_id: str, mode: str, scope: str) -> Optional[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, video_id, mode, scope, source_hash, status,
                          audio_url, duration_seconds, provider, source_label,
                          error_message, metadata, created_at, updated_at
                   FROM audio_artifacts
                   WHERE video_id = %s AND mode = %s AND scope = %s""",
                [video_id, mode, scope],
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ---- Write ----

    def upsert_artifact(self, video_id: str, mode: str, scope: str,
                        source_hash: str, status: str = "queued",
                        **kwargs) -> int:
        """Insert or update an artifact row. Returns the row id."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO audio_artifacts
                       (video_id, mode, scope, source_hash, status,
                        audio_url, duration_seconds, provider, source_label,
                        error_message, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (video_id, mode, scope)
                   DO UPDATE SET
                       source_hash = EXCLUDED.source_hash,
                       status = EXCLUDED.status,
                       audio_url = EXCLUDED.audio_url,
                       duration_seconds = EXCLUDED.duration_seconds,
                       provider = EXCLUDED.provider,
                       source_label = EXCLUDED.source_label,
                       error_message = EXCLUDED.error_message,
                       metadata = EXCLUDED.metadata,
                       updated_at = NOW()
                   RETURNING id""",
                [video_id, mode, scope, source_hash, status,
                 kwargs.get("audio_url"),
                 kwargs.get("duration_seconds"),
                 kwargs.get("provider"),
                 kwargs.get("source_label"),
                 kwargs.get("error_message"),
                 _json_dumps(kwargs.get("metadata"))],
            )
            row = cur.fetchone()
            conn.commit()
            return row.get("id") if row else None

    def update_status(self, video_id: str, mode: str, scope: str,
                      status: str, **kwargs) -> None:
        """Update status and optional fields on an existing artifact."""
        sets = ["status = %s", "updated_at = NOW()"]
        params: list = [status]

        for field in ("audio_url", "duration_seconds", "provider",
                       "source_label", "error_message"):
            if field in kwargs:
                sets.append(f"{field} = %s")
                params.append(kwargs[field])

        params.extend([video_id, mode, scope])

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""UPDATE audio_artifacts SET {', '.join(sets)}
                    WHERE video_id = %s AND mode = %s AND scope = %s""",
                params,
            )
            conn.commit()

    # ---- Source text resolution ----

    def resolve_source_text(self, video_id: str, scope: str,
                            variant_slug: Optional[str] = None) -> str:
        """Get canonical source text from DB for a given scope."""
        with self._connect() as conn, conn.cursor() as cur:
            if scope == "summary_active":
                if variant_slug:
                    cur.execute(
                        """SELECT text FROM v_latest_summaries
                           WHERE video_id = %s AND variant = %s LIMIT 1""",
                        [video_id, variant_slug],
                    )
                else:
                    cur.execute(
                        """SELECT text FROM v_latest_summaries
                           WHERE video_id = %s
                           AND variant NOT LIKE 'audio%%'
                           AND variant != 'deep-research'
                           ORDER BY variant LIMIT 1""",
                        [video_id],
                    )
                row = cur.fetchone()
                return row.get("text", "") if row else ""

            elif scope == "ponderings_visible":
                cur.execute(
                    """SELECT research_response FROM follow_up_research_runs
                       WHERE video_id = %s
                       ORDER BY created_at DESC LIMIT 1""",
                    [video_id],
                )
                row = cur.fetchone()
                return row.get("research_response", "") if row else ""

            elif scope == "transcript_visible":
                cur.execute(
                    "SELECT transcript_text FROM content WHERE video_id = %s",
                    [video_id],
                )
                row = cur.fetchone()
                return row.get("transcript_text", "") if row else ""

            return ""

    def resolve_source_label(self, video_id: str, scope: str,
                             variant_slug: Optional[str] = None) -> str:
        """Get a human-readable label for the source."""
        with self._connect() as conn, conn.cursor() as cur:
            if scope == "summary_active" and variant_slug:
                return variant_slug.replace("-", " ").title()
            elif scope == "ponderings_visible":
                return "Research Report"
            elif scope == "transcript_visible":
                return "Transcript"
            return "Summary"


def compute_source_hash(mode: str, scope: str, source_text: str) -> str:
    """Compute SHA-256 hash for cache validation."""
    canonical = (mode + ":" + scope + ":" + (source_text or "")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_dumps(val) -> Optional[str]:
    if val is None:
        return None
    import json
    return json.dumps(val)

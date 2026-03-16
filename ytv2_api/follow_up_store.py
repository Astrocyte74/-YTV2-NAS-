"""Persistence helpers for follow-up research API endpoints."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore

from modules.summary_variants import format_summary_html

logger = logging.getLogger(__name__)


def _env_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or "postgres"
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or ""
    dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "postgres"
    sslmode = os.getenv("PGSSLMODE", "require")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{dbname}?sslmode={sslmode}"


def _infer_source_type(url: str) -> str:
    lower = (url or "").lower()
    if "youtube.com" in lower or "youtu.be" in lower:
        return "youtube"
    if "reddit.com" in lower or "redd.it" in lower:
        return "reddit"
    if lower:
        return "web"
    return "unknown"


def _candidate_video_ids(video_id: str, source_type: str | None = None) -> list[str]:
    candidates: list[str] = []

    def _add(value: str) -> None:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    _add(video_id)
    if ":" in (video_id or ""):
        _add(video_id.split(":", 1)[1])
        return candidates

    normalized_source = (source_type or "").strip().lower()
    alias_prefix = {
        "youtube": "yt",
        "reddit": "reddit",
        "web": "web",
    }.get(normalized_source)
    if alias_prefix:
        _add(f"{alias_prefix}:{video_id}")
    else:
        for prefix in ("yt", "reddit", "web"):
            _add(f"{prefix}:{video_id}")
    return candidates


def _coerce_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


@dataclass
class ResolvedFollowUpContext:
    video_id: str
    summary_id: int | None
    summary: str
    source_context: dict[str, Any]


class FollowUpStore:
    """Persistence layer for follow-up research and summary references."""

    def __init__(self, dsn: str | None = None):
        global psycopg
        if psycopg is None:
            try:
                import importlib
                psycopg = importlib.import_module("psycopg")
            except Exception as exc:
                raise RuntimeError(
                    "psycopg is not installed. Please add 'psycopg[binary]>=3.1' to requirements.txt."
                ) from exc
        self.dsn = dsn or _env_dsn()

    def _connect(self):
        return psycopg.connect(self.dsn)

    def resolve_context(
        self,
        *,
        video_id: str,
        summary_id: int | None = None,
        summary: str = "",
        source_context: dict[str, Any] | None = None,
        preferred_variant: str | None = None,
    ) -> ResolvedFollowUpContext:
        """Resolve summary metadata from Postgres and merge with request context."""
        merged_context = dict(source_context or {})
        source_type = str(merged_context.get("type") or "").strip().lower() or None
        row = self._fetch_summary_row(
            video_id=video_id,
            summary_id=summary_id,
            preferred_variant=preferred_variant,
            source_type=source_type,
        )

        if row is not None:
            resolved_video_id = str(row["video_id"])
            if summary_id is not None and resolved_video_id != video_id:
                raise LookupError(f"Summary {summary_id} does not belong to video_id {video_id}")
            merged_context.setdefault("title", row.get("title") or "")
            merged_context.setdefault("url", row.get("canonical_url") or "")
            merged_context.setdefault("published_at", row.get("published_at"))
            merged_context.setdefault("type", _infer_source_type(str(row.get("canonical_url") or "")))
            merged_context.setdefault("video_id", resolved_video_id)
            merged_context.setdefault("id", resolved_video_id)
            return ResolvedFollowUpContext(
                video_id=resolved_video_id,
                summary_id=int(row["id"]),
                summary=(row.get("text") or "").strip(),
                source_context=merged_context,
            )

        merged_context.setdefault("video_id", video_id)
        merged_context.setdefault("id", video_id)
        merged_context.setdefault("type", _infer_source_type(str(merged_context.get("url") or "")))
        return ResolvedFollowUpContext(
            video_id=video_id,
            summary_id=summary_id,
            summary=(summary or "").strip(),
            source_context=merged_context,
        )

    def _fetch_summary_row(
        self,
        *,
        video_id: str,
        summary_id: int | None,
        preferred_variant: str | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            row = None
            if summary_id is not None:
                cur.execute(
                    """
                    SELECT s.id, s.video_id, s.text, s.variant, c.title, c.canonical_url, c.published_at
                    FROM summaries s
                    LEFT JOIN content c ON c.video_id = s.video_id
                    WHERE s.id = %s
                    LIMIT 1
                    """,
                    (summary_id,),
                )
                row = cur.fetchone()
            elif preferred_variant:
                candidate_ids = _candidate_video_ids(video_id, source_type)
                placeholders = ", ".join(["%s"] * len(candidate_ids))
                cur.execute(
                    f"""
                    SELECT s.id, s.video_id, s.text, s.variant, c.title, c.canonical_url, c.published_at
                    FROM summaries s
                    LEFT JOIN content c ON c.video_id = s.video_id
                    WHERE s.video_id IN ({placeholders})
                      AND COALESCE(s.text, '') <> ''
                      AND s.variant = %s
                    ORDER BY
                      s.created_at DESC,
                      COALESCE(s.revision, 1) DESC,
                      s.id DESC
                    LIMIT 1
                    """,
                    (*candidate_ids, preferred_variant),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        f"""
                        SELECT s.id, s.video_id, s.text, s.variant, c.title, c.canonical_url, c.published_at
                        FROM summaries s
                        LEFT JOIN content c ON c.video_id = s.video_id
                        WHERE s.video_id IN ({placeholders})
                          AND COALESCE(s.text, '') <> ''
                          AND s.variant NOT LIKE 'audio%%'
                          AND s.variant <> 'deep-research'
                        ORDER BY
                          CASE
                            WHEN s.variant = 'comprehensive' THEN 0
                            WHEN s.variant = 'bullet-points' THEN 1
                            WHEN s.variant = 'key-insights' THEN 2
                            ELSE 10
                          END,
                          s.created_at DESC,
                          COALESCE(s.revision, 1) DESC,
                          s.id DESC
                        LIMIT 1
                        """,
                        tuple(candidate_ids),
                    )
                    row = cur.fetchone()
            else:
                candidate_ids = _candidate_video_ids(video_id, source_type)
                placeholders = ", ".join(["%s"] * len(candidate_ids))
                cur.execute(
                    f"""
                    SELECT s.id, s.video_id, s.text, s.variant, c.title, c.canonical_url, c.published_at
                    FROM summaries s
                    LEFT JOIN content c ON c.video_id = s.video_id
                    WHERE s.video_id IN ({placeholders})
                      AND COALESCE(s.text, '') <> ''
                      AND s.variant NOT LIKE 'audio%%'
                      AND s.variant <> 'deep-research'
                    ORDER BY
                      CASE
                        WHEN s.variant = 'comprehensive' THEN 0
                        WHEN s.variant = 'bullet-points' THEN 1
                        WHEN s.variant = 'key-insights' THEN 2
                        ELSE 10
                      END,
                      s.created_at DESC,
                      COALESCE(s.revision, 1) DESC,
                      s.id DESC
                    LIMIT 1
                    """,
                    tuple(candidate_ids),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "video_id": row[1],
            "text": row[2],
            "variant": row[3],
            "title": row[4],
            "canonical_url": row[5],
            "published_at": row[6].isoformat() if isinstance(row[6], datetime) else row[6],
        }

    def store_suggestions(
        self,
        *,
        video_id: str,
        summary_id: int,
        suggestions: list[dict[str, Any]],
        planner_provider: str | None = None,
        planner_model: str | None = None,
    ) -> None:
        safe_provider = str(planner_provider or "unknown").strip() or "unknown"
        safe_model = str(planner_model or "unknown").strip() or "unknown"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO follow_up_suggestions (
                  video_id, summary_id, suggestions, generated_at, planner_provider, planner_model, expires_at
                )
                VALUES (%s, %s, %s::jsonb, now(), %s, %s, now() + INTERVAL '7 days')
                ON CONFLICT (summary_id) DO UPDATE SET
                  suggestions = EXCLUDED.suggestions,
                  generated_at = EXCLUDED.generated_at,
                  planner_provider = EXCLUDED.planner_provider,
                  planner_model = EXCLUDED.planner_model,
                  expires_at = EXCLUDED.expires_at
                """,
                (video_id, summary_id, json.dumps(suggestions), safe_provider, safe_model),
            )
            conn.commit()

    def get_stored_suggestions(self, summary_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT suggestions
                FROM follow_up_suggestions
                WHERE summary_id = %s
                LIMIT 1
                """,
                (summary_id,),
            )
            row = cur.fetchone()
        if row is None:
            return []
        suggestions = _coerce_json(row[0], [])
        return suggestions if isinstance(suggestions, list) else []

    def get_cached_research(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, video_id, summary_id, research_response, research_meta, status, cache_key
                FROM follow_up_research_runs
                WHERE cache_key = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (cache_key,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        meta = _coerce_json(row[4], {})
        return {
            "run_id": row[0],
            "video_id": row[1],
            "summary_id": row[2],
            "answer": row[3] or "",
            "meta": meta,
            "status": row[5] or "ok",
            "cache_key": row[6],
            "sources": list(meta.get("stored_sources") or []),
        }

    def store_research_run(
        self,
        *,
        video_id: str,
        summary_id: int,
        approved_questions: list[str],
        question_provenance: list[str] | None,
        result,
    ) -> int:
        meta = dict(result.meta or {})
        meta["stored_sources"] = [
            {
                "name": source.name,
                "url": source.url,
                "domain": source.domain,
                "tier": source.tier,
                "providers": list(source.providers),
                "tools": list(source.tools),
            }
            for source in result.sources
        ]

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO follow_up_research_runs (
                  video_id,
                  summary_id,
                  approved_questions,
                  question_provenance,
                  question_kinds,
                  planned_queries,
                  coverage_map,
                  research_response,
                  research_meta,
                  cache_key,
                  created_at,
                  completed_at,
                  status
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, now(), now(), %s
                )
                ON CONFLICT (cache_key) DO UPDATE SET
                  approved_questions = EXCLUDED.approved_questions,
                  question_provenance = EXCLUDED.question_provenance,
                  question_kinds = EXCLUDED.question_kinds,
                  planned_queries = EXCLUDED.planned_queries,
                  coverage_map = EXCLUDED.coverage_map,
                  research_response = EXCLUDED.research_response,
                  research_meta = EXCLUDED.research_meta,
                  completed_at = EXCLUDED.completed_at,
                  status = EXCLUDED.status
                RETURNING id
                """,
                (
                    video_id,
                    summary_id,
                    approved_questions,
                    question_provenance or list(meta.get("question_provenance") or []),
                    list(meta.get("question_kinds") or []),
                    list(meta.get("planned_queries") or []),
                    json.dumps(meta.get("coverage_map") or []),
                    result.answer,
                    json.dumps(meta),
                    meta.get("cache_key"),
                    result.status,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        return int(row[0])

    def create_summary_variant_reference(self, *, video_id: str, text: str) -> dict[str, int]:
        reference_text = "Follow-up research available. Canonical result stored in follow_up_research_runs."
        html = format_summary_html(reference_text)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM summaries WHERE video_id = %s AND variant = 'deep-research'",
                (video_id,),
            )
            revision = int(cur.fetchone()[0])
            cur.execute(
                """
                INSERT INTO summaries (video_id, variant, revision, text, html, created_at)
                VALUES (%s, 'deep-research', %s, %s, %s, now())
                RETURNING id, revision
                """,
                (video_id, revision, reference_text, html),
            )
            row = cur.fetchone()
            conn.commit()
        return {"summary_variant_id": int(row[0]), "summary_variant_revision": int(row[1])}

    def mark_follow_up_available(self, summary_id: int) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE summaries SET follow_up_research_available = TRUE WHERE id = %s",
                (summary_id,),
            )
            conn.commit()


__all__ = ["FollowUpStore", "ResolvedFollowUpContext"]

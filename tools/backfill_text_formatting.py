#!/usr/bin/env python3
"""
Backfill minimal HTML for text summary variants across recent items.

- Queries recent video_ids from content (ordered by indexed_at desc)
- For each video_id, fetches latest rows from summaries by variant
- For each text variant (comprehensive, key-insights, bullet-points, executive):
    * If text exists, rebuild HTML via modules.summary_variants.format_summary_html
    * If --dry-run, print a short diff signal and continue
    * If --apply, insert a new revision with rebuilt HTML (do not modify audio)

Usage examples:
  python3 tools/backfill_text_formatting.py --limit 10 --dry-run
  python3 tools/backfill_text_formatting.py --limit 10 --apply
  python3 tools/backfill_text_formatting.py --video-id web:abcd1234 --apply

Notes:
  - Only allowed tags/classes are produced by the formatter: p, h3.kp-heading,
    ul.kp-list, li, strong, em, a (href sanitized by dashboard)
  - Existing revisions are not modified — we insert a new revision per variant
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.summary_variants import format_summary_html  # noqa: E402


TEXT_VARIANTS = ("comprehensive", "key-insights", "bullet-points", "executive")


def fetch_recent_video_ids(conn: psycopg.Connection, *, limit: int, offset: int = 0) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT video_id
            FROM content
            ORDER BY indexed_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_latest_by_variant(conn: psycopg.Connection, video_id: str) -> Dict[str, Tuple[Optional[str], Optional[str], int]]:
    """Return {variant: (text, html, revision)} for latest rows per variant."""
    sql = (
        "SELECT DISTINCT ON (variant) variant, text, html, revision "
        "FROM summaries WHERE video_id=%s ORDER BY variant, created_at DESC, revision DESC"
    )
    result: Dict[str, Tuple[Optional[str], Optional[str], int]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (video_id,))
        for variant, text, html, rev in cur.fetchall():
            result[str(variant)] = (text, html, int(rev or 0))
    return result


def next_revision(conn: psycopg.Connection, video_id: str, variant: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(revision),0) FROM summaries WHERE video_id=%s AND variant=%s", (video_id, variant))
        (rev,) = cur.fetchone()
        return int(rev) + 1


def insert_revision(conn: psycopg.Connection, video_id: str, variant: str, text: str, html: str, revision: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO summaries (video_id, variant, revision, text, html) VALUES (%s,%s,%s,%s,%s)",
            (video_id, variant, revision, text, html),
        )
    conn.commit()


def needs_reformat(text: Optional[str], html: Optional[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if not html or not str(html).strip():
        return True
    shtml = str(html)
    # Heuristics: no headings or lists in HTML but text has bullets/markdown headings
    if ("<h3" not in shtml and "<ul" not in shtml) and (
        "•" in text or re.search(r"(?m)^[\-*]\s", text) or re.search(r"(?m)^(?:#{1,6}|\*\*).+", text)
    ):
        return True
    # Additional cue: text has markdown-style headings but HTML has no <h3>
    if re.search(r"(?m)^(?:#{1,6}\s+|\*\*).+", text) and "<h3" not in shtml:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill minimal HTML for text variants")
    ap.add_argument("--video-id", help="Single video_id to process")
    ap.add_argument("--limit", type=int, default=10, help="Recent items to scan when no video-id")
    ap.add_argument("--offset", type=int, default=0, help="Offset into recent items when no video-id (for pagination)")
    ap.add_argument("--variants", default="all", help="Comma list or 'all' (comprehensive,key-insights,bullet-points,executive)")
    ap.add_argument("--apply", action="store_true", help="Write new revisions; otherwise dry-run")
    ap.add_argument("--dry-run", action="store_true", help="Alias for default dry-run; if both set, --apply wins")
    ap.add_argument("--force", action="store_true", help="Force reformat even if heuristics consider HTML acceptable")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set")
        return 2

    if args.variants.strip().lower() == "all":
        variants = TEXT_VARIANTS
    else:
        variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())

    with psycopg.connect(dsn) as conn:
        video_ids: List[str]
        if args.video_id:
            video_ids = [args.video_id]
        else:
            video_ids = fetch_recent_video_ids(conn, limit=args.limit, offset=args.offset)

        changed = 0
        scanned = 0
        for vid in video_ids:
            latest = fetch_latest_by_variant(conn, vid)
            for var in variants:
                if var.startswith("audio"):
                    continue
                text, html, _ = latest.get(var, (None, None, 0))
                scanned += 1
                if not needs_reformat(text, html):
                    continue
                new_html = format_summary_html(text or "")
                if not new_html or (html and str(html).strip() == new_html.strip()):
                    continue
                changed += 1
                if args.apply and not args.dry_run:
                    rev = next_revision(conn, vid, var)
                    insert_revision(conn, vid, var, text or "", new_html, rev)
                    print(f"APPLIED {vid} {var} -> revision {rev} ({len(new_html)} chars)")
                else:
                    print(f"DRYRUN {vid} {var} -> would insert new html ({len(new_html)} chars)")

        print(f"Done. scanned={scanned} changed={changed} apply={bool(args.apply and not args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

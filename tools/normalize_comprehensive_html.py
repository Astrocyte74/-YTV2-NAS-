#!/usr/bin/env python3
"""
Normalize legacy 'comprehensive' summaries by promoting headings and cleaning bullets.

- Converts existing HTML into plain text with heading lines and bullets.
- Rebuilds semantic minimal HTML via modules.summary_variants.format_summary_html.
- Inserts a new summaries revision (append-only) for variant='comprehensive'.

Usage:
  python3 tools/normalize_comprehensive_html.py --video-id <id> [--video-id <id> ...] [--dry-run]

Notes:
  - Safe, append-only. Original revisions are preserved.
  - Only touches 'comprehensive' variant.
"""

from __future__ import annotations

import argparse
import html as htmlmod
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import psycopg

# Ensure project imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.summary_variants import format_summary_html  # type: ignore


def fetch_latest_comprehensive(conn: psycopg.Connection, video_id: str) -> Tuple[Optional[str], Optional[str], int]:
    sql = (
        "SELECT text, html, COALESCE(revision,1) FROM summaries "
        "WHERE video_id=%s AND variant='comprehensive' "
        "ORDER BY created_at DESC, revision DESC LIMIT 1"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (video_id,))
        row = cur.fetchone()
        if not row:
            return None, None, 0
        text, html, rev = row
        return (text if isinstance(text, str) else None,
                html if isinstance(html, str) else None,
                int(rev or 0))


def next_revision(conn: psycopg.Connection, video_id: str, variant: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(revision),0) FROM summaries WHERE video_id=%s AND variant=%s", (video_id, variant))
        (rev,) = cur.fetchone()
        return int(rev) + 1


def insert_revision(conn: psycopg.Connection, video_id: str, text: str, html: str, revision: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO summaries (video_id, variant, revision, text, html) VALUES (%s,'comprehensive',%s,%s,%s)",
            (video_id, revision, text, html),
        )
    conn.commit()


def html_to_text_for_normalize(html: str) -> str:
    """Convert legacy HTML into plain text for formatter.

    - Promote <div class="kp-heading"> and paragraphs starting with '##'/'###' to heading lines
    - Convert <li> to '- ' lines
    - Paragraphs to blank lines; strip tags; unescape; strip markdown wrappers
    """
    s = html or ""
    # Normalize line breaks
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)

    def _emit_heading(m):
        t = htmlmod.unescape(m.group(1))
        # strip leading hashes or trailing hashes
        t = re.sub(r"^#{1,6}\s+", "", t)
        t = re.sub(r"\s+#{1,6}$", "", t)
        # strip markdown bold wrappers
        t = re.sub(r"^(\*\*|__)(.+?)(\*\*|__)\s*:?$", r"\2", t)
        return f"{t.strip()}:\n"

    # Promote div.kp-heading
    s = re.sub(r"<div[^>]*class=\"kp-heading\"[^>]*>(.*?)</div>", _emit_heading, s, flags=re.IGNORECASE|re.DOTALL)
    # Promote paragraphs that contain ATX headings
    s = re.sub(r"<p[^>]*>\s*(###+\s+[^<]+)\s*</p>", _emit_heading, s, flags=re.IGNORECASE)
    s = re.sub(r"<p[^>]*>\s*(##\s+[^<]+)\s*</p>", _emit_heading, s, flags=re.IGNORECASE)

    # Convert list items to '- ' lines
    s = re.sub(r"<li[^>]*>\s*", "- ", s, flags=re.IGNORECASE)
    s = re.sub(r"</li>", "\n", s, flags=re.IGNORECASE)

    # Paragraphs → blanks
    s = re.sub(r"</p>", "\n\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<p[^>]*>", "", s, flags=re.IGNORECASE)

    # Strip remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # Unescape entities
    s = htmlmod.unescape(s)
    # Strip markdown bold/italic wrappers in residual text
    s = re.sub(r"(\*\*|__)(.+?)(\*\*|__)", r"\2", s)
    s = re.sub(r"(\*|_)(.+?)(\*|_)", r"\2", s)
    # Collapse whitespace
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize legacy 'comprehensive' HTML and insert new revision")
    ap.add_argument("--video-id", action="append", required=True, help="Video id (repeat for multiple)")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing to DB")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set")
        return 2

    ids = []
    for v in args.video_id:
        parts = [p.strip() for p in (v.split(',') if ',' in v else [v])]
        ids.extend([p for p in parts if p])

    changed = 0
    with psycopg.connect(dsn) as conn:
        for vid in ids:
            text, html, _ = fetch_latest_comprehensive(conn, vid)
            if not html and not text:
                print(f"SKIP {vid} no comprehensive variant")
                continue
            # Prefer using HTML to derive clean text, since legacy issues are in HTML
            source = html or text or ""
            derived_text = html_to_text_for_normalize(source)
            new_html = format_summary_html(derived_text)
            if not new_html:
                print(f"SKIP {vid} could not build new html")
                continue
            if html and html.strip() == new_html.strip():
                print(f"SKIP {vid} unchanged")
                continue
            if args.dry_run:
                print(f"DRYRUN {vid} comprehensive -> would insert new html ({len(new_html)} chars)")
                changed += 1
                continue
            rev = next_revision(conn, vid, 'comprehensive')
            insert_revision(conn, vid, derived_text, new_html, rev)
            print(f"APPLIED {vid} comprehensive -> revision {rev} ({len(new_html)} chars)")
            changed += 1

    print(f"Done. changed={changed} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


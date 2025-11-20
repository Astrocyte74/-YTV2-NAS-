#!/usr/bin/env python3
"""
Utility to inspect (and optionally clear) summary_image_url / summary_image_ai2_url fields.

Typical use: after deleting a batch of PNGs from the dashboard, run this script with the
same filename substring or timestamp and pass --apply once you confirm the selection.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List content rows whose summary images match the supplied filters and optionally"
            " null out their summary_image_url / analysis_json->summary_image_ai2_url fields."
        )
    )
    parser.add_argument(
        "--after",
        metavar="UTC_TIMESTAMP",
        help="Only target rows with updated_at >= this UTC timestamp (e.g. 2025-11-17T00:00:00).",
    )
    parser.add_argument(
        "--contains",
        metavar="SUBSTRING",
        action="append",
        help="Match rows where either image URL contains this substring (can be repeated).",
    )
    parser.add_argument(
        "--ids-file",
        metavar="PATH",
        help="Optional file containing one content.id per line to scope the update.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many matching rows to display (applies before --apply).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually clear summary_image_url / summary_image_ai2_url for the matched rows.",
    )
    return parser.parse_args()


def load_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def format_row(row: Tuple[str, str, str | None, str | None, datetime]) -> str:
    ident, title, img1, img2, updated = row
    return f"{updated:%Y-%m-%d %H:%M:%S} | {ident} | {title[:80]} | {img1 or '-'} | {img2 or '-'}"


def main() -> None:
    args = parse_args()

    db_url = (
        os.getenv("DATABASE_URL_POSTGRES_NEW")
        or os.getenv("DATABASE_URL")
        or os.getenv("DATABASE_URL_POSTGRES_INTERNAL")
    )
    if not db_url:
        raise SystemExit("Set DATABASE_URL_POSTGRES_NEW (or DATABASE_URL) before running this script.")

    rows: Sequence[Tuple[str, str, str | None, str | None, datetime]]

    where_parts: List[str] = [
        "(summary_image_url IS NOT NULL OR (analysis_json->>'summary_image_ai2_url') IS NOT NULL)"
    ]
    params: List[object] = []

    if args.after:
        where_parts.append("updated_at >= %s")
        params.append(args.after)

    if args.contains:
        for token in args.contains:
            like = f"%{token}%"
            where_parts.append(
                "(COALESCE(summary_image_url, '') ILIKE %s OR COALESCE(analysis_json->>'summary_image_ai2_url', '') ILIKE %s)"
            )
            params.extend([like, like])

    id_list: List[str] = []
    if args.ids_file:
        id_list = load_ids(args.ids_file)
        if not id_list:
            raise SystemExit(f"No ids found in {args.ids_file}")
        where_parts.append("id = ANY(%s)")
        params.append(id_list)

    if len(where_parts) == 1:
        raise SystemExit("Refusing to run without at least one filter (--after/--contains/--ids-file).")

    where_clause = " AND ".join(where_parts)
    limit_sql = " LIMIT %s" if args.limit else ""
    if args.limit:
        params_with_limit = [*params, args.limit]
    else:
        params_with_limit = params

    query = f"""
        SELECT id, title, summary_image_url, analysis_json->>'summary_image_ai2_url', updated_at
        FROM content
        WHERE {where_clause}
        ORDER BY updated_at DESC{limit_sql}
    """

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params_with_limit)
            rows = cur.fetchall()

    if not rows:
        print("No matching rows.")
        return

    print("Matched rows:")
    for row in rows:
        print(" -", format_row(row))

    if not args.apply:
        print(f"\n{len(rows)} rows would be cleared. Re-run with --apply to update.")
        return

    ids = [row[0] for row in rows]
    update_sql = """
        UPDATE content
        SET summary_image_url = NULL,
            analysis_json = CASE
                WHEN analysis_json IS NULL THEN NULL
                ELSE jsonb_strip_nulls(
                    jsonb_set(analysis_json, '{summary_image_ai2_url}', 'null'::jsonb, true)
                )
            END
        WHERE id = ANY(%s)
    """

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(update_sql, (ids,))
        conn.commit()

    print(f"Cleared summary_image fields for {len(ids)} rows.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backfill 'key-insights' summaries from existing Key‑Points (flat bullets) using local Ollama models.

Two modes:
  - Dry-run (default): generate candidate 'key-insights' for each model and save under
    exports/experiments/insights/<timestamp>/<video_id>/ without touching Postgres.
  - Apply: choose a winner model and insert a new 'summaries' revision (variant='key-insights').

Source input per video_id:
  - Prefer the latest Key‑Points (bullet-points/key-points) text.
  - If available, the script may also load a cached report JSON (data/reports/*video-id*) to use transcript
    in future, but by default we convert flat bullets into thematic 'Key‑Insights'.

Usage examples:
  python3 tools/backfill_insights_from_keypoints.py --limit 10 --models "ollama:gemma3:12b,ollama:gpt-oss:20b-cloud" --dry-run
  python3 tools/backfill_insights_from_keypoints.py --limit 10 --models "ollama:gemma3:12b,ollama:gpt-oss:20b-cloud" --apply --winner first

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg

# Ensure imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import ollama_client as oc  # type: ignore
from modules.summary_variants import format_summary_html  # type: ignore


TEXT_VARIANTS = ("bullet-points", "key-points")


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def build_prompt_from_bullets(bullets_text: str) -> str:
    return (
        "You are converting flat Key-Points into Key-Insights.\n\n"
        "Input is a plain list of bullets and maybe an overview + Bottom line.\n"
        "Your job: group the bullets into 3–5 short thematic sections, each with 3–5 concise bullets.\n"
        "Do not invent facts; only restructure what’s present. Preserve language and meaning.\n"
        "Do not add a global Bottom line.\n\n"
        "Output format (text only):\n"
        "### <Short Heading>\n- bullet\n- bullet\n- bullet\n\n"
        "Use '### ' for headings and '- ' for bullets. No code fences, no emojis.\n\n"
        f"Key-Points:\n{bullets_text.strip()}\n"
    )


def call_ollama_model(model: str, prompt: str) -> Optional[str]:
    try:
        res = oc.generate(prompt=prompt, model=model, stream=False)
        txt = (res or {}).get("response") or (res or {}).get("message", {}).get("content")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return None
    except Exception as e:
        print(f"LLM error for {model}: {e}")
        return None


def prefer_bullet_points_text(latest: Dict[str, Tuple[Optional[str], Optional[str], int]]) -> Optional[str]:
    # Try 'bullet-points' then 'key-points'
    for v in ("bullet-points", "key-points"):
        t = latest.get(v, (None, None, 0))[0]
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def load_report_text(video_id: str) -> Optional[str]:
    # Fallback to cached report JSON under data/reports
    root = Path("data/reports")
    if not root.is_dir():
        return None
    hint = video_id.split(":")[-1]
    matches = [p for p in root.glob("*.json") if hint in p.name]
    for p in sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # If the report already has a key-insights summary text, we can use that as a base
            summ = data.get("summary") or {}
            t = summ.get("summary")
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill Key-Insights from Key-Points using local Ollama models")
    ap.add_argument("--video-id", help="Single video_id to process")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--models", default="ollama:gemma3:12b,ollama:gpt-oss:20b-cloud", help="Comma-separated provider:model list (ollama only)")
    ap.add_argument("--apply", action="store_true", help="Insert 'key-insights' revision with winner model output")
    ap.add_argument("--winner", default="first", help="Winner model to write when --apply: first|gemma|oss20b|model-slug")
    ap.add_argument("--only-missing", action="store_true", help="Skip items that already have key-insights")
    ap.add_argument("--out-dir", help="Where to store dry-run outputs (defaults to exports/experiments/insights/<ts>)")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set")
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ts = now_ts()
    out_root = Path(args.out_dir) if args.out_dir else (Path("exports/experiments/insights") / ts)
    out_root.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn) as conn:
        if args.video_id:
            video_ids = [args.video_id]
        else:
            video_ids = fetch_recent_video_ids(conn, limit=args.limit, offset=args.offset)

        processed = 0
        applied = 0
        for vid in video_ids:
            latest = fetch_latest_by_variant(conn, vid)
            # Require Key-Points present
            kp_text = prefer_bullet_points_text(latest)
            if not kp_text:
                continue
            # Optionally skip if key-insights already exists
            if args.only-missing and latest.get("key-insights", (None, None, 0))[0]:
                continue

            prompt = build_prompt_from_bullets(kp_text)
            exp_dir = out_root / vid.replace(":", "_")
            exp_dir.mkdir(parents=True, exist_ok=True)

            model_outputs: Dict[str, Dict[str, str]] = {}
            for spec in models:
                provider, model = (spec.split(":", 1) + [""])[:2]
                if provider != "ollama":
                    print(f"Skipping non-ollama provider for {vid}: {spec}")
                    continue
                txt = call_ollama_model(model, prompt)
                if not txt:
                    continue
                html = format_summary_html(txt)
                # Save artifacts (dry-run always saves; on apply also save for audit)
                (exp_dir / f"{model.replace('/', '_').replace(':', '-')}.md").write_text(txt, encoding="utf-8")
                (exp_dir / f"{model.replace('/', '_').replace(':', '-')}.html").write_text(html, encoding="utf-8")
                model_outputs[model] = {"text": txt, "html": html}

            # Choose winner if applying
            if args.apply and model_outputs:
                winner_key = args.winner.strip().lower()
                ordered = [spec.split(":", 1)[1] for spec in models if spec.startswith("ollama:")]
                if winner_key == "first" and ordered:
                    pick = ordered[0]
                elif winner_key in ("gemma", "gemma3", "gemma3:12b"):
                    pick = "gemma3:12b"
                elif winner_key in ("oss20b", "gpt-oss:20b-cloud"):
                    pick = "gpt-oss:20b-cloud"
                else:
                    pick = winner_key

                chosen = model_outputs.get(pick)
                if not chosen:
                    # fallback to first available
                    pick = next(iter(model_outputs.keys()))
                    chosen = model_outputs[pick]

                rev = next_revision(conn, vid, "key-insights")
                insert_revision(conn, vid, "key-insights", chosen["text"], chosen["html"], rev)
                applied += 1
                print(f"APPLIED {vid} key-insights <- {pick} (revision {rev})")

            processed += 1

        print(f"Done. processed={processed} applied={applied} out={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


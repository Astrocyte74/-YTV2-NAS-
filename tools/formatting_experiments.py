#!/usr/bin/env python3
"""
Run formatting prompt experiments across multiple LLM models (local/cloud).

Inputs
- A plain-text summary input (via --text-file or --text)
- One or more models specified as provider:model (e.g., ollama:phi3:latest)
- One or more formats to test: markdown, html

Outputs
- Writes model outputs to exports/experiments/formatting/<timestamp>/
- Saves per-run JSON metadata (latency, lengths) and a manifest.json

Providers
- Ollama (local) via hub or direct using modules.ollama_client
- OpenAI (requires OPENAI_API_KEY)
- OpenRouter (requires OPENROUTER_API_KEY)

Notes
- This script does live calls. If keys or local hub are missing, runs are skipped.
- Designed for quick, reproducible comparisons without changing core prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests

# Ensure project root is on sys.path so "modules" is importable when invoked as tools/...
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prefer local client for ollama
try:
    from modules import ollama_client
except Exception:
    ollama_client = None  # type: ignore


def read_text_input(args: argparse.Namespace) -> str:
    # DB-sourced overrides file/text when specified
    if args.video_id or args.pick_longest:
        db_txt = fetch_text_from_db(video_id=args.video_id, pick_longest=args.pick_longest, min_chars=args.min_chars)
        if not db_txt:
            raise RuntimeError("No summary text found in DB for the given criteria")
        return db_txt
    if args.text:
        return args.text.strip()
    if args.text_file:
        p = Path(args.text_file)
        if not p.is_file():
            raise FileNotFoundError(f"--text-file not found: {p}")
        return p.read_text(encoding="utf-8").strip()
    # Fallback sample
    sample = (
        "Main topic: Using small language models for daily workflows.\n"
        "Key points:\n"
        "- Local models are fast and private, suitable for formatting tasks.\n"
        "- Markdown output is compact and easy to render safely.\n"
        "- HTML can be used with a small, whitelisted tag set.\n"
        "Takeaway: Prefer deterministic post-formatting or Markdown + sanitizer."
    )
    return sample


def build_prompt(fmt: str, content: str) -> Tuple[str, str]:
    """Return (prompt, expected_ext) for the requested format."""
    if fmt == "markdown":
        prompt = f"""
You are a careful formatter. Reformat the given content into clean GitHub-flavored Markdown.

Requirements:
- Preserve every fact, number, name, order, and language.
- Use headings and lists where appropriate (## for section headings; - for bullets).
- Do not add commentary, code fences, or emojis.
- Output only the Markdown. No preamble or explanation.

Content:
"""
        prompt = prompt.strip() + "\n\n" + content.strip()
        return prompt, "md"

    if fmt == "html":
        prompt = f"""
You are a careful formatter. Reformat the given content into a minimal, valid HTML snippet.

Requirements:
- Preserve every fact, number, name, order, and language.
- Use only these tags: <p>, <h3>, <ul>, <li>, <strong>, <em>, <a>.
- No other tags, no inline styles, no scripts.
- Prefer an opening paragraph, then <h3> sections with <ul><li> bullets when appropriate.
- For a final conclusion, use a paragraph starting with <strong>Bottom line:</strong>.
- Output only the HTML snippet. No preamble or explanation.

Content:
"""
        prompt = prompt.strip() + "\n\n" + content.strip()
        return prompt, "html"

    raise ValueError(f"Unsupported format: {fmt}")


def call_ollama(model: str, prompt: str) -> str:
    if ollama_client is None:
        raise RuntimeError("modules.ollama_client not available")
    # Use non-stream generate for simple prompt tests
    res = ollama_client.generate(prompt=prompt, model=model, stream=False)
    # Hub/direct wrappers return {'response': '...'}
    txt = (res or {}).get("response") or (res or {}).get("message", {}).get("content")
    if not isinstance(txt, str):
        raise RuntimeError(f"Unexpected Ollama response: {res}")
    return txt


def call_openai(model: str, prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    txt = ((choice.get("message") or {}).get("content") or "").strip()
    return txt


def call_openrouter(model: str, prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://ytv2.local/"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "YTV2 Formatting Experiments"),
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    txt = ((choice.get("message") or {}).get("content") or "").strip()
    return txt


def parse_model_spec(spec: str) -> Tuple[str, str]:
    # provider:model (model may contain colons or slashes)
    if ":" not in spec:
        raise ValueError(f"Model spec must be provider:model, got '{spec}'")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    return provider, model


def fetch_text_from_db(*, video_id: Optional[str] = None, pick_longest: bool = False, min_chars: int = 1000) -> Optional[str]:
    """Fetch summary text from Postgres.

    - When video_id is provided, prefer 'comprehensive', else longest text among standard variants.
    - When pick_longest=True, return the single longest summary text above min_chars across all videos.
    Requires DATABASE_URL and psycopg.
    """
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return None
    try:
        import psycopg
    except Exception:
        return None

    variants = ("comprehensive", "bullet-points", "key-insights")
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                if video_id:
                    cur.execute(
                        """
                        SELECT text
                        FROM summaries
                        WHERE video_id = %s
                          AND variant = ANY(%s)
                          AND text IS NOT NULL
                        ORDER BY (variant='comprehensive') DESC, created_at DESC, char_length(text) DESC
                        LIMIT 1
                        """,
                        (video_id, list(variants)),
                    )
                    row = cur.fetchone()
                    if row and isinstance(row[0], str):
                        return row[0].strip()
                    # Fallback: any text for the video
                    cur.execute(
                        """
                        SELECT text
                        FROM summaries
                        WHERE video_id = %s AND text IS NOT NULL
                        ORDER BY char_length(text) DESC, created_at DESC
                        LIMIT 1
                        """,
                        (video_id,),
                    )
                    row = cur.fetchone()
                    return (row[0].strip() if row and isinstance(row[0], str) else None)

                if pick_longest:
                    cur.execute(
                        """
                        SELECT text
                        FROM summaries
                        WHERE text IS NOT NULL AND variant = ANY(%s)
                        ORDER BY char_length(text) DESC, created_at DESC
                        LIMIT 1
                        """,
                        (list(variants),),
                    )
                    row = cur.fetchone()
                    if row and isinstance(row[0], str) and len(row[0]) >= min_chars:
                        return row[0].strip()
                    # Fallback: any variant
                    cur.execute(
                        """
                        SELECT text
                        FROM summaries
                        WHERE text IS NOT NULL
                        ORDER BY char_length(text) DESC, created_at DESC
                        LIMIT 1
                        """
                    )
                    row = cur.fetchone()
                    return (row[0].strip() if row and isinstance(row[0], str) else None)
    except Exception:
        return None
    return None


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Run LLM formatting experiments")
    ap.add_argument("--text-file", help="Path to input plain-text summary")
    ap.add_argument("--text", help="Inline text content (overrides --text-file)")
    ap.add_argument(
        "--formats",
        default="markdown,html",
        help="Comma-separated formats to test: markdown,html",
    )
    ap.add_argument(
        "--models",
        default="ollama:phi3:latest,ollama:gemma3:12b",
        help="Comma-separated model specs: provider:model",
    )
    ap.add_argument("--video-id", help="Fetch summary text from DB for this video_id")
    ap.add_argument("--pick-longest", action="store_true", help="Use the longest available summary text from DB")
    ap.add_argument("--min-chars", type=int, default=1000, help="Minimum chars when --pick-longest is used")
    ap.add_argument("--out-dir", help="Output directory (default under exports/experiments/formatting)")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts and exit")
    args = ap.parse_args(argv)

    content = read_text_input(args)
    formats = [f.strip().lower() for f in (args.formats or "").split(",") if f.strip()]
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]

    # Prepare output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out_dir) if args.out_dir else Path("exports/experiments/formatting") / ts
    base_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": ts,
        "formats": formats,
        "models": models,
        "runs": [],
    }

    for fmt in formats:
        prompt, ext = build_prompt(fmt, content)
        if args.dry_run:
            print("===== FORMAT:", fmt)
            print(prompt)
            print()
            continue
        for spec in models:
            provider, model = parse_model_spec(spec)
            label = f"{provider}__{model.replace('/', '_').replace(':', '-') }__{fmt}"
            started = time.time()
            output_text = None
            error = None
            try:
                if provider == "ollama":
                    output_text = call_ollama(model, prompt)
                elif provider == "openai":
                    output_text = call_openai(model, prompt)
                elif provider == "openrouter":
                    output_text = call_openrouter(model, prompt)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
            except Exception as e:
                error = str(e)
            elapsed = round(time.time() - started, 3)

            run_meta = {
                "provider": provider,
                "model": model,
                "format": fmt,
                "seconds": elapsed,
            }

            if output_text and not error:
                out_path = base_out / f"{label}.{ext}"
                out_path.write_text(output_text, encoding="utf-8")
                run_meta.update({
                    "ok": True,
                    "output_file": str(out_path),
                    "chars": len(output_text),
                    "lines": output_text.count("\n") + 1,
                })
            else:
                run_meta.update({"ok": False, "error": error or "unknown"})

            # Write per-run metadata
            (base_out / f"{label}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
            manifest["runs"].append(run_meta)

    # Save manifest
    (base_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        return 0

    # Print a compact summary
    ok = sum(1 for r in manifest["runs"] if r.get("ok"))
    total = len(manifest["runs"]) or 1
    print(f"Completed runs: {ok}/{total} ok. Outputs in {base_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

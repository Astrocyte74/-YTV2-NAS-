#!/usr/bin/env python3
"""
Quick probe for YouTube transcript availability and 429 status using youtube-transcript-api.

Usage examples:
  python3 tools/test_youtube_transcripts.py --ids dQw4w9WgXcQ,9bZkp7q19f0 --repeat 2 --sleep 1.5
  python3 tools/test_youtube_transcripts.py --from-file ids.txt --sleep 2 --jitter 0.5

The script:
  - Attempts to fetch English (or auto) transcripts for each video id
  - Repeats the pass `--repeat` times with pacing (`--sleep` + random jitter)
  - Reports counts for OK / no transcript / disabled / 429 / other errors

Exit code: 0 on success; 1 if any 429s observed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (  # type: ignore
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        CouldNotRetrieveTranscript,
    )
except Exception as e:  # pragma: no cover
    print("youtube-transcript-api is required. pip install youtube-transcript-api", file=sys.stderr)
    raise


def parse_ids(raw: str) -> List[str]:
    ids: List[str] = []
    for token in (raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        # Allow full URLs; extract v param or last path token when matching YouTube forms
        if "youtube.com" in t or "youtu.be" in t:
            import urllib.parse as up
            try:
                u = up.urlparse(t)
                if u.netloc.endswith("youtu.be"):
                    cand = u.path.strip("/")
                    if cand:
                        ids.append(cand)
                        continue
                q = up.parse_qs(u.query)
                v = (q.get("v") or [""])[0]
                if v:
                    ids.append(v)
                    continue
            except Exception:
                pass
        ids.append(t)
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def load_ids_from_file(path: Path) -> List[str]:
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return parse_ids(",".join(raw))


_API_INSTANCE = None
try:
    _API_INSTANCE = YouTubeTranscriptApi()
except Exception:
    _API_INSTANCE = None


def try_get_transcript(video_id: str, languages: List[str]) -> Tuple[str, int]:
    """Return (status, length) where status in {OK, NO_TRANSCRIPT, DISABLED, UNAVAILABLE, RETRIEVE_ERROR, ERR_429, ERROR}."""
    try:
        # Prefer en; allow auto-generated variants
        # This call raises errors we catch below on failure cases
        # Support multiple variants across versions
        api = _API_INSTANCE
        tr_items = []
        if api and hasattr(api, "list"):
            listing = api.list(video_id)
            transcript = None
            try:
                transcript = listing.find_transcript(languages)
            except Exception:
                try:
                    transcript = listing.find_generated_transcript(languages)
                except Exception:
                    transcript = None
            if not transcript:
                return "NO_TRANSCRIPT", 0
            tr_items = transcript.fetch()
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            listing = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
            transcript = None
            try:
                transcript = listing.find_transcript(languages)
            except Exception:
                try:
                    transcript = listing.find_generated_transcript(languages)
                except Exception:
                    transcript = None
            if not transcript:
                return "NO_TRANSCRIPT", 0
            tr_items = transcript.fetch()
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            tr_items = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
        elif api and hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages)
            tr_items = list(fetched)
        else:
            return "RETRIEVE_ERROR", 0

        text = " ".join(item.get("text", "") for item in (tr_items or []))
        return "OK", len(text)
    except TranscriptsDisabled:
        return "DISABLED", 0
    except NoTranscriptFound:
        return "NO_TRANSCRIPT", 0
    except VideoUnavailable:
        return "UNAVAILABLE", 0
    except CouldNotRetrieveTranscript as e:  # often wraps HTTP errors
        msg = str(e).lower()
        if "429" in msg or "too many requests" in msg:
            return "ERR_429", 0
        return "RETRIEVE_ERROR", 0
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "too many requests" in msg:
            return "ERR_429", 0
        return "ERROR", 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe YouTube transcripts for 429s")
    ap.add_argument("--ids", help="Comma-separated video ids or URLs")
    ap.add_argument("--from-file", dest="from_file", help="File with video ids/URLs, one per line")
    ap.add_argument("--sleep", type=float, default=1.5, help="Base sleep seconds between requests")
    ap.add_argument("--jitter", type=float, default=0.5, help="Random jitter added to sleep (0..jitter)")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat passes over list this many times")
    ap.add_argument("--lang", default="en", help="Preferred language code (comma-separated to try multiple)")
    args = ap.parse_args()

    ids: List[str] = []
    if args.ids:
        ids.extend(parse_ids(args.ids))
    if args.from_file:
        ids.extend(load_ids_from_file(Path(args.from_file)))
    ids = [i for i in ids if i]
    if not ids:
        # A tiny default set for quick checks
        ids = [
            "dQw4w9WgXcQ",  # Rick Astley
            "9bZkp7q19f0",  # PSY - GANGNAM STYLE
            "3JZ_D3ELwOQ",  # Adele - Hello
        ]

    langs = [s.strip() for s in (args.lang or "en").split(",") if s.strip()]
    langs = langs or ["en"]

    print(f"Testing {len(ids)} ids x {args.repeat} passes; sleep={args.sleep}s jitter<= {args.jitter}s; langs={langs}")
    counts = {k: 0 for k in ("OK", "NO_TRANSCRIPT", "DISABLED", "UNAVAILABLE", "RETRIEVE_ERROR", "ERR_429", "ERROR")}

    for r in range(args.repeat):
        print(f"\nPass {r+1}/{args.repeat}")
        for vid in ids:
            status, length = try_get_transcript(vid, langs)
            counts[status] = counts.get(status, 0) + 1
            if status == "OK":
                print(f"  {vid}: OK ({length} chars)")
            else:
                print(f"  {vid}: {status}")
            delay = max(0.0, args.sleep + random.uniform(0.0, max(0.0, args.jitter)))
            time.sleep(delay)

    print("\nSummary:")
    for k in ("OK", "NO_TRANSCRIPT", "DISABLED", "UNAVAILABLE", "RETRIEVE_ERROR", "ERR_429", "ERROR"):
        print(f"  {k:15s} {counts.get(k, 0)}")

    return 1 if counts.get("ERR_429", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Quick tester for YouTube Data API v3 (videos.list) using an API key from env.

Validates that API calls succeed from this environment/IP and prints key fields.

Usage:
  python3 tools/test_youtube_data_api.py --ids dQw4w9WgXcQ,9bZkp7q19f0
  python3 tools/test_youtube_data_api.py --from-file ids.txt

Env required:
  - YT_API_KEY (string)

Notes:
  - Each videos.list call costs 1 unit; this tool uses a single batched call.
  - API key should be restricted to this IP and to YouTube Data API v3 only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import requests


def parse_ids(raw: str) -> List[str]:
    vids: List[str] = []
    for tok in (raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        if "youtube.com" in t or "youtu.be" in t:
            # extract v param or short id
            from urllib.parse import urlparse, parse_qs
            u = urlparse(t)
            if u.netloc.endswith("youtu.be"):
                short = u.path.strip("/")
                if short:
                    vids.append(short)
                    continue
            v = (parse_qs(u.query).get("v") or [""])[0]
            if v:
                vids.append(v)
                continue
        vids.append(t)
    # dedupe preserving order
    out: List[str] = []
    seen = set()
    for v in vids:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def load_ids(path: Path) -> List[str]:
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return parse_ids(",".join(raw))


def iso8601_duration_to_seconds(dur: str) -> int:
    # Minimal ISO 8601 duration parser for PT#H#M#S
    import re
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", dur or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s


def main() -> int:
    ap = argparse.ArgumentParser(description="Test YouTube Data API v3 videos.list")
    ap.add_argument("--ids", help="Comma-separated video IDs or URLs")
    ap.add_argument("--from-file", dest="from_file", help="File with video IDs/URLs, one per line")
    args = ap.parse_args()

    key = os.getenv("YT_API_KEY")
    if not key:
        print("YT_API_KEY not set in env", file=sys.stderr)
        return 2

    vids: List[str] = []
    if args.ids:
        vids.extend(parse_ids(args.ids))
    if args.from_file:
        vids.extend(load_ids(Path(args.from_file)))
    vids = [v for v in vids if v]
    if not vids:
        vids = ["dQw4w9WgXcQ"]

    # Batch into a single request (API supports up to 50 IDs per call)
    ids_param = ",".join(vids[:50])
    url = (
        "https://www.googleapis.com/youtube/v3/videos?"
        f"part=snippet,contentDetails,statistics&id={ids_param}&key={key}"
    )

    print(f"Calling YouTube Data API for {len(vids[:50])} id(s)…")
    try:
        r = requests.get(url, timeout=12)
    except Exception as e:
        print(f"Request error: {e}", file=sys.stderr)
        return 1

    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:800])
        return 1

    data = r.json() or {}
    items = data.get("items") or []
    if not items:
        print("No items returned (check API key restrictions, IP, or quota).")
        return 1

    for it in items:
        vid = it.get("id")
        sn = it.get("snippet") or {}
        cd = it.get("contentDetails") or {}
        st = it.get("statistics") or {}
        title = (sn.get("title") or "").strip()
        chan = (sn.get("channelTitle") or "").strip()
        dur = cd.get("duration") or ""
        secs = iso8601_duration_to_seconds(dur)
        views = st.get("viewCount")
        print(f"- {vid}: '{title}' | channel='{chan}' | duration={secs}s | views={views}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""
Quick tester for the Wikipedia REST fast-path used by WebPageExtractor.

Usage:
  python3 tools/test_wikipedia_fastpath.py --url https://en.wikipedia.org/wiki/Raid_at_Cabanatuan --mode auto

Modes:
  auto | full | summary | off
"""

from __future__ import annotations

import argparse
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.web_extract import WebPageExtractor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Test Wikipedia fast-path extraction")
    ap.add_argument("--url", required=True, help="Wikipedia article URL")
    ap.add_argument("--mode", default="auto", help="auto|full|summary|off (default auto)")
    args = ap.parse_args()

    os.environ["WIKI_API_MODE"] = args.mode
    ex = WebPageExtractor()
    content = ex.extract(args.url)

    print("Title:", content.title)
    print("Canonical:", content.canonical_url)
    print("Language:", content.language)
    print("Notes:", content.extractor_notes)
    print("Text chars:", len(content.text))
    print("HTML chars:", len(content.html or ""))
    print("Preview:", (content.text or "").split("\n")[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

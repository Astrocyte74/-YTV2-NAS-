#!/usr/bin/env python3
"""
Run Wikipedia sectioned Insights on a single URL using the current wikiBranch pipeline.

Usage:
  WIKI_API_MODE=auto python3 tools/run_wiki_insights.py --url "https://en.wikipedia.org/wiki/Raid_at_Cabanatuan" \
      --model "gemma3:12b" --provider ollama

Env knobs:
  - WIKI_API_MODE: auto|full|summary|off (default auto)
  - WIKI_SECTION_LIMIT: default 8
  - WIKI_SECTION_MAX_CHARS: default 2200
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from youtube_summarizer import YouTubeSummarizer  # noqa: E402


async def _run(url: str, provider: str, model: str) -> int:
    try:
        ys = YouTubeSummarizer(llm_provider=provider, model=model)
    except Exception as e:
        print("Summarizer init error:", e)
        return 2
    result = await ys.process_web_page(url, summary_type="key-insights")

    if isinstance(result, dict) and result.get("summary"):
        summ = result["summary"]
        headline = summ.get("headline") if isinstance(summ, dict) else None
        text = summ.get("summary") if isinstance(summ, dict) else str(summ)
        print("\n=== HEADLINE ===\n", headline or "(none)")
        print("\n=== INSIGHTS ===\n", text)
        return 0
    print("No summary generated:", result)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Wikipedia Insights pipeline on a URL")
    ap.add_argument("--url", required=True)
    ap.add_argument("--provider", default=os.getenv("WIKI_TEST_PROVIDER", "ollama"))
    ap.add_argument("--model", default=os.getenv("WIKI_TEST_MODEL", "gemma3:12b"))
    args = ap.parse_args()
    return asyncio.run(_run(args.url, args.provider, args.model))


if __name__ == "__main__":
    raise SystemExit(main())


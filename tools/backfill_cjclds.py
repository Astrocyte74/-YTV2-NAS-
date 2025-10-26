#!/usr/bin/env python3
"""
Backfill CJCLDS category for existing reports (NAS)

Scans /app/data/reports for JSON reports, detects General Conference talks
from churchofjesuschrist.org, adds CJCLDS + speaker subcategory, and upserts
to the dashboard via the existing dual-sync path.

Usage examples:
  python3 tools/backfill_cjclds.py --dry-run
  python3 tools/backfill_cjclds.py --limit 50
  python3 tools/backfill_cjclds.py --only-host churchofjesuschrist.org
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from modules.cjclds import classify_and_apply_cjclds
from modules.services import sync_service

def _resolve_reports_dir() -> Path:
    # Prefer container path
    p = Path("/app/data/reports")
    if p.exists():
        return p
    # Fallback to workspace paths
    for candidate in (Path("./data/reports"), Path("data/reports")):
        if candidate.exists():
            return candidate
    return p  # default (may not exist)

REPORTS_DIR = _resolve_reports_dir()


def _get_url(report: Dict[str, Any]) -> Optional[str]:
    # Prefer canonical_url in known locations
    url = (
        report.get("canonical_url")
        or (report.get("source_metadata", {}).get("web", {}) or {}).get("canonical_url")
        or (report.get("source_metadata", {}).get("youtube", {}) or {}).get("canonical_url")
        or (report.get("metadata", {}) or {}).get("url")
    )
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def _has_cjclds(report: Dict[str, Any]) -> bool:
    sc = report.get("subcategories_json")
    if not isinstance(sc, dict):
        return False
    cats = sc.get("categories")
    if not isinstance(cats, list):
        return False
    for entry in cats:
        if not isinstance(entry, dict):
            continue
        if (entry.get("category") or "").strip().lower() == "cjclds":
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill CJCLDS category for existing reports")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files or sync; just report candidates")
    parser.add_argument("--limit", type=int, default=0, help="Max number of updates to apply (0=no limit)")
    parser.add_argument("--only-host", type=str, default="churchofjesuschrist.org", help="Host to restrict matches to")
    args = parser.parse_args()

    if not REPORTS_DIR.exists():
        print(f"Reports directory missing: {REPORTS_DIR}")
        return 1

    files = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    inspected = 0
    eligible = 0
    updated = 0
    synced = 0

    for path in files:
        inspected += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️  Skip unreadable JSON: {path.name} ({exc})")
            continue

        url = _get_url(data)
        if args.only_host and (not url or args.only_host not in url):
            continue

        new_data = classify_and_apply_cjclds(dict(data), url)

        # If no change, continue
        if json.dumps(new_data, sort_keys=True) == json.dumps(data, sort_keys=True):
            continue

        eligible += 1
        print(f"📝 Update will be applied: {path.name}")
        if args.dry_run:
            continue

        # Write back
        try:
            path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
        except Exception as exc:
            print(f"❌ Failed to write {path.name}: {exc}")
            continue

        # Sync content only (report path is sufficient; no audio path)
        try:
            result = sync_service.run_dual_sync(path, label=path.stem)
            if isinstance(result, dict) and result.get("success"):
                synced += 1
                print(f"✅ Synced: {path.name}")
            else:
                print(f"⚠️  Sync failed: {path.name}")
        except Exception as exc:
            print(f"❌ Sync error: {path.name}: {exc}")

        if args.limit and synced >= args.limit:
            break

    print("\nSummary:")
    print(f"  Inspected: {inspected}")
    print(f"  Eligible:  {eligible}")
    print(f"  Updated:   {updated}")
    print(f"  Synced:    {synced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

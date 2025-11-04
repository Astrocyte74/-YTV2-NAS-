#!/usr/bin/env python3
from __future__ import annotations

"""
Quick dashboard health probe from NAS.

Prints /api/version and /api/health/storage (if DASHBOARD_DEBUG_TOKEN is set).
Exits non-zero if storage used_pct exceeds a threshold (default 98%).

Usage:
  python3 tools/dashboard_health.py [--block-pct 98]
"""

import argparse
import os
import sys
from typing import Any

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("requests is required") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-pct", type=int, default=int(os.getenv("DASHBOARD_STORAGE_BLOCK_PCT") or 98))
    args = ap.parse_args()

    base = (os.getenv("RENDER_DASHBOARD_URL") or os.getenv("RENDER_API_URL") or "").rstrip("/")
    if not base:
        print("RENDER_DASHBOARD_URL is required", file=sys.stderr)
        return 2

    try:
        r = requests.get(f"{base}/api/version", timeout=10)
        print("version:", r.status_code, r.text)
    except Exception as exc:  # pragma: no cover
        print("version_error:", exc, file=sys.stderr)

    tok = os.getenv("DASHBOARD_DEBUG_TOKEN")
    if not tok:
        print("storage: token not set; skipping gated probe")
        return 0

    try:
        h = requests.get(f"{base}/api/health/storage", headers={"Authorization": f"Bearer {tok}"}, timeout=10)
        print("storage:", h.status_code)
        if h.status_code == 200:
            data: dict[str, Any] = h.json()
            used_pct = int(data.get("used_pct") or 0)
            print({"used_pct": used_pct, "free_bytes": data.get("free_bytes"), "total_bytes": data.get("total_bytes")})
            if used_pct >= args.block_pct:
                print(f"CRITICAL: used_pct={used_pct} >= {args.block_pct}")
                return 1
        else:
            print(h.text)
    except Exception as exc:  # pragma: no cover
        print("storage_error:", exc, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


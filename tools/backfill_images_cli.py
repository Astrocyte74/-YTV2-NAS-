#!/usr/bin/env python3
"""
Interactive helper for running tools/backfill_summary_images.py.
Prompts for the common knobs (limit, mode, plan-only/dry-run, etc.)
and then invokes the existing backfill script with those options.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKFILL_SCRIPT = PROJECT_ROOT / "tools" / "backfill_summary_images.py"


def prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            return int(raw)
        try:
            return int(float(raw))
        except Exception:
            print("Please enter a number.")


def prompt_bool(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} ({suffix}): ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


def prompt_choice(prompt: str, choices: List[str], default: str) -> str:
    choice_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} ({choice_str}) [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"Please choose one of: {choice_str}")


def prompt_list(prompt: str) -> List[str]:
    raw = input(f"{prompt} (comma-separated, blank for none): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    if not BACKFILL_SCRIPT.exists():
        print(f"Backfill script not found: {BACKFILL_SCRIPT}")
        return 1

    print("Summary Image Backfill CLI")
    print("---------------------------")

    limit = prompt_int("How many rows to process", 25)
    mode = prompt_choice("Image mode", ["ai1", "ai2"], "ai1")
    plan_only = prompt_bool("Plan only (list targets, no generation)?", default=True)
    dry_run = False
    if not plan_only:
        dry_run = prompt_bool("Dry run (generate but skip upload/DB)?", default=False)
    only_missing_thumbnail = prompt_bool("Only rows missing thumbnail_url?", default=False)
    delay = prompt_int("Delay between jobs (ms)", 200) / 1000.0
    video_ids = prompt_list("Specific video IDs (optional)")

    cmd: List[str] = [
        sys.executable,
        str(BACKFILL_SCRIPT),
        "--limit",
        str(limit),
        "--mode",
        mode,
        "--delay",
        str(delay),
    ]
    if plan_only:
        cmd.append("--plan-only")
    if dry_run:
        cmd.append("--dry-run")
    if only_missing_thumbnail:
        cmd.append("--only-missing-thumbnail")
    for vid in video_ids:
        cmd.extend(["--video-id", vid])

    print("\nAbout to run:")
    print(" ", " ".join(shlex.quote(part) for part in cmd))
    if not prompt_bool("Proceed?", default=True):
        print("Aborted.")
        return 0

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Simple helper to inspect Reddit env vars inside the container.

Loads .env/.env.nas so values managed in files are visible.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

KEYS = (
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_REFRESH_TOKEN",
    "REDDIT_USER_AGENT",
)


def _load_env_files():
    if not load_dotenv:
        return
    load_dotenv(dotenv_path=Path('.env'), override=False)
    load_dotenv(dotenv_path=Path('.env.nas'), override=True)


def main() -> None:
    _load_env_files()
    for key in KEYS:
        value = os.getenv(key)
        if value is None:
            print(f"{key}: MISSING")
        else:
            # Show trimmed info (common issue is stray whitespace)
            trimmed = value.strip()
            note = " (trimmed)" if trimmed != value else ""
            display = repr(trimmed)
            print(f"{key}: len={len(trimmed)} repr={display}{note}")


if __name__ == "__main__":
    main()

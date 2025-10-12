#!/usr/bin/env python3
"""Simple helper to inspect Reddit env vars inside the container."""

import os

KEYS = (
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_REFRESH_TOKEN",
    "REDDIT_USER_AGENT",
)


def main() -> None:
    for key in KEYS:
        value = os.getenv(key)
        if value is None:
            print(f"{key}: MISSING")
        else:
            display = repr(value)
            print(f"{key}: len={len(value)} repr={display}")


if __name__ == "__main__":
    main()

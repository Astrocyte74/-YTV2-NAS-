#!/usr/bin/env python3
"""
Quick smoke test for Reddit API credentials.

Run this inside the NAS container:
    python tools/test_reddit_connection.py <reddit_post_url>

This script loads .env/.env.nas automatically so you can manage
credentials in files rather than Portainer env vars.

Example URL:
    https://www.reddit.com/r/Python/comments/1f9vmxq/whats_everyone_working_on_this_week/

It verifies required environment variables, logs in via PRAW, and prints
basic details about the submission plus the top-level comment.
"""

import os
import sys

import praw
from pathlib import Path
import requests
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


REDDIT_REQUIRED_VARS = ("REDDIT_CLIENT_ID", "REDDIT_REFRESH_TOKEN", "REDDIT_USER_AGENT")
REDDIT_OPTIONAL_VARS = ("REDDIT_CLIENT_SECRET",)

def _load_env_files():
    """Load .env and .env.nas if python-dotenv is available."""
    if not load_dotenv:
        return
    # Project defaults first, then NAS overrides
    load_dotenv(dotenv_path=Path('.env'), override=False)
    load_dotenv(dotenv_path=Path('.env.nas'), override=True)

def _env_stripped(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip()


def ensure_env():
    """Abort with a clear message if any required Reddit env var is missing."""
    missing = [key for key in REDDIT_REQUIRED_VARS if not os.getenv(key)]
    if missing:
        for key in missing:
            print(f"❌ Missing environment variable: {key}")
        sys.exit(1)


def build_client() -> praw.Reddit:
    """Instantiate a PRAW client using refresh-token auth."""
    return praw.Reddit(
        client_id=_env_stripped("REDDIT_CLIENT_ID"),
        client_secret=_env_stripped("REDDIT_CLIENT_SECRET"),
        refresh_token=_env_stripped("REDDIT_REFRESH_TOKEN"),
        user_agent=_env_stripped("REDDIT_USER_AGENT"),
    )


def parse_args():
    """Parse CLI flags."""
    args = sys.argv[1:]
    login_only = False
    url = None

    if "--login-only" in args:
        login_only = True
        args.remove("--login-only")

    if args:
        url = args.pop(0)

    if args:
        print("Usage: python tools/test_reddit_connection.py [--login-only] <reddit_post_url>")
        sys.exit(1)

    if not login_only and not url:
        try:
            url = input("Reddit post URL: ").strip()
        except EOFError:
            url = ""
        if not url:
            print("Usage: python tools/test_reddit_connection.py [--login-only] <reddit_post_url>")
            sys.exit(1)

    return login_only, url


def main() -> None:
    _load_env_files()
    login_only, url = parse_args()

    ensure_env()

    reddit = build_client()
    me = reddit.user.me()
    print(f"✅ Authenticated as: {me}")

    if login_only:
        return

    # Resolve share/short links to canonical URL first
    try:
        resp = requests.get(url, allow_redirects=True, timeout=8)
        url = resp.url or url
    except Exception:
        pass

    submission = reddit.submission(url=url)
    try:
        submission.comments.replace_more(limit=0)
        comments = submission.comments.list()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"\n⚠️ Unable to load comments: {exc}")
        comments = []

    try:
        title = submission.title
        body = submission.selftext
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"\n⚠️ Unable to load submission body: {exc}")
        title = None
        body = ""

    if title:
        print("\n🔗 Title:", title)
    else:
        print("\n🔗 Title: <unavailable>")

    if body:
        print("📝 Body preview:", body[:300], "..." if len(body) > 300 else "")
    else:
        print("📝 Body preview: <no selftext>")

    if comments:
        print("\n💬 Top comment preview:", comments[0].body[:300], "..." if len(comments[0].body) > 300 else "")
    else:
        print("\n💬 Top comment preview: <no comments>")


if __name__ == "__main__":
    main()

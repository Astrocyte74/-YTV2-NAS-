#!/usr/bin/env python3
"""
Quick smoke test for Reddit API credentials.

Run this inside the NAS container (env populated via .env.nas):
    python tools/test_reddit_connection.py <reddit_post_url>

Example URL:
    https://www.reddit.com/r/Python/comments/1f9vmxq/whats_everyone_working_on_this_week/

The script verifies the required environment variables, logs in via PRAW,
and prints basic details about the submission plus the top-level comment.
"""

import os
import sys

import praw


REDDIT_REQUIRED_VARS = ("REDDIT_CLIENT_ID", "REDDIT_REFRESH_TOKEN", "REDDIT_USER_AGENT")
REDDIT_OPTIONAL_VARS = ("REDDIT_CLIENT_SECRET",)


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
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
        refresh_token=os.environ["REDDIT_REFRESH_TOKEN"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
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
    login_only, url = parse_args()

    ensure_env()

    reddit = build_client()
    me = reddit.user.me()
    print(f"✅ Authenticated as: {me}")

    if login_only:
        return

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

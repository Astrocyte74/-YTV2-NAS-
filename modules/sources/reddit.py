"""
Reddit content fetcher for YTV2.

Provides a thin wrapper around PRAW that fetches a submission and returns
normalized metadata plus a combined text payload suitable for summarization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import praw
from praw.models import Comment
from prawcore import PrawcoreException


class RedditFetcherError(Exception):
    """Raised when Reddit content cannot be fetched or normalized."""


def _env(name: str) -> Optional[str]:
    """Fetch an environment variable and return ``None`` if blank."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass
class RedditFetchResult:
    """Structured Reddit submission payload."""

    id: str
    title: str
    author: str
    subreddit: str
    url: str
    created_utc: float
    score: int
    upvote_ratio: float
    num_comments: int
    flair: Optional[str]
    thumbnail: Optional[str]
    selftext: str
    comment_snippets: List[Dict[str, Any]] = field(default_factory=list)
    combined_text: str = ""
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "subreddit": self.subreddit,
            "canonical_url": self.url,
            "created_utc": self.created_utc,
            "created_at_iso": datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat(),
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "flair": self.flair,
            "thumbnail": self.thumbnail,
            "selftext": self.selftext,
            "comment_snippets": self.comment_snippets,
            "combined_text": self.combined_text,
            "language": self.language or "en",
        }


class RedditFetcher:
    """Fetches Reddit submissions using refresh-token OAuth."""

    DEFAULT_COMMENT_LIMIT = 12
    PER_COMMENT_CHAR_LIMIT = 600
    MAX_COMBINED_CHARS = 15000

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or _env("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or _env("REDDIT_CLIENT_SECRET") or ""
        self.refresh_token = refresh_token or _env("REDDIT_REFRESH_TOKEN")
        self.user_agent = user_agent or _env("REDDIT_USER_AGENT")

        missing = [
            name
            for name, value in [
                ("REDDIT_CLIENT_ID", self.client_id),
                ("REDDIT_REFRESH_TOKEN", self.refresh_token),
                ("REDDIT_USER_AGENT", self.user_agent),
            ]
            if not value
        ]
        if missing:
            raise RedditFetcherError(
                f"Missing Reddit credentials: {', '.join(missing)}. "
                "Set them in the environment before processing Reddit URLs."
            )

        try:
            self._client = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret or "",
                refresh_token=self.refresh_token,
                user_agent=self.user_agent,
            )
        except Exception as exc:  # pragma: no cover - PRAW handles errors internally
            raise RedditFetcherError(f"Failed to initialize Reddit client: {exc}") from exc

    @staticmethod
    def _clean_thumbnail(thumbnail: Optional[str]) -> Optional[str]:
        if not thumbnail:
            return None
        thumbnail = thumbnail.strip()
        if thumbnail in {"self", "default", "image", "spoiler", "nsfw"}:
            return None
        return thumbnail

    def fetch(self, url: str, comment_limit: Optional[int] = None) -> RedditFetchResult:
        """Fetch a Reddit submission and return normalized content."""
        if not url:
            raise RedditFetcherError("Empty Reddit URL provided.")

        comment_limit = comment_limit or self.DEFAULT_COMMENT_LIMIT

        try:
            submission = self._client.submission(url=url)
        except Exception as exc:
            raise RedditFetcherError(f"Unable to load submission: {exc}") from exc

        try:
            submission.comments.replace_more(limit=1)
        except PrawcoreException as exc:
            raise RedditFetcherError(f"Reddit API error while loading comments: {exc}") from exc

        snippets: List[Dict[str, Any]] = []
        for top_level in submission.comments:
            if len(snippets) >= comment_limit:
                break
            if not isinstance(top_level, Comment):
                continue
            body = (top_level.body or "").strip()
            if not body or body == "[deleted]" or body == "[removed]":
                continue
            if getattr(top_level, "stickied", False):
                continue
            snippet_text = body[: self.PER_COMMENT_CHAR_LIMIT].strip()
            if len(body) > self.PER_COMMENT_CHAR_LIMIT:
                snippet_text += "…"
            snippets.append(
                {
                    "id": top_level.id,
                    "author": (top_level.author.name if top_level.author else "[deleted]"),
                    "score": int(top_level.score or 0),
                    "body": snippet_text,
                }
            )

        comments_block = ""
        if snippets:
            formatted = []
            for item in snippets:
                author = item["author"]
                score = item["score"]
                text = item["body"]
                formatted.append(f"Comment by u/{author} (score {score}):\n{text}")
            comments_block = "\n\n".join(formatted)

        parts = [
            submission.title or "",
            submission.selftext or "",
        ]
        if comments_block:
            parts.append("Top comments:\n" + comments_block)

        combined_text = "\n\n".join(part for part in parts if part).strip()
        if len(combined_text) > self.MAX_COMBINED_CHARS:
            combined_text = (
                combined_text[: self.MAX_COMBINED_CHARS].rstrip() + "\n\n[Truncated for length]"
            )

        author_name = submission.author.name if submission.author else "[deleted]"
        permalink = getattr(submission, "permalink", "")
        canonical_url = f"https://www.reddit.com{permalink}" if permalink else url

        return RedditFetchResult(
            id=submission.id,
            title=submission.title or "Untitled thread",
            author=author_name,
            subreddit=submission.subreddit.display_name if submission.subreddit else "unknown",
            url=canonical_url,
            created_utc=float(submission.created_utc or datetime.now(timezone.utc).timestamp()),
            score=int(submission.score or 0),
            upvote_ratio=float(submission.upvote_ratio or 0.0),
            num_comments=int(submission.num_comments or 0),
            flair=submission.link_flair_text,
            thumbnail=self._clean_thumbnail(getattr(submission, "thumbnail", None)),
            selftext=submission.selftext or "",
            comment_snippets=snippets,
            combined_text=combined_text,
            language=getattr(submission, "lang", None),
        )

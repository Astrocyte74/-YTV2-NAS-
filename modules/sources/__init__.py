"""Content source fetchers for YTV2."""

__all__ = []

try:
    from .reddit import RedditFetcher, RedditFetcherError

    __all__.extend(["RedditFetcher", "RedditFetcherError"])
except Exception:  # pragma: no cover - optional dependency
    RedditFetcher = None

    class RedditFetcherError(Exception):
        """Placeholder error when Reddit fetcher is unavailable."""

try:
    from .web import WebPageFetcher, WebFetcherError

    __all__.extend(["WebPageFetcher", "WebFetcherError"])
except Exception:  # pragma: no cover - optional dependency
    WebPageFetcher = None

    class WebFetcherError(Exception):
        """Placeholder error when web fetcher is unavailable."""

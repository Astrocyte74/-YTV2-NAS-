"""Configuration for the portable research service."""

from __future__ import annotations

import os


def _env_str(name: str, default: str = "", *, legacy: str | None = None) -> str:
    if legacy and os.environ.get(legacy) is not None and os.environ.get(name) is None:
        return str(os.environ.get(legacy, default) or default)
    return str(os.environ.get(name, default) or default)


def _env_bool(name: str, default: bool, *, legacy: str | None = None) -> bool:
    raw_default = "true" if default else "false"
    raw_value = _env_str(name, raw_default, legacy=legacy).strip().lower()
    return raw_value in {"1", "true", "yes"}


RESEARCH_ENABLED = _env_bool("RESEARCH_ENABLED", True, legacy="IMAGE_RESEARCH_ENABLED")
INCEPTION_API_KEY = _env_str("INCEPTION_API_KEY")
INCEPTION_URL = _env_str("INCEPTION_URL", "https://api.inceptionlabs.ai/v1/chat/completions")
INCEPTION_MODEL = _env_str("INCEPTION_MODEL", _env_str("CHAT_INTENT_MODEL", "mercury-2"))

RESEARCH_FALLBACK_ENABLED = _env_bool("RESEARCH_FALLBACK_ENABLED", False)
RESEARCH_FALLBACK_MODEL = _env_str("RESEARCH_FALLBACK_MODEL", "google/gemini-3.1-flash-lite-preview")
OPENROUTER_API_KEY = _env_str("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HTTP_REFERER = _env_str("OPENROUTER_HTTP_REFERER", "https://research-service.local")
OPENROUTER_APP_TITLE = _env_str("OPENROUTER_APP_TITLE", "Portable Research Service")

RESEARCH_PLANNER_PROVIDER = _env_str("RESEARCH_PLANNER_PROVIDER", "auto").strip().lower()
if RESEARCH_PLANNER_PROVIDER not in {"auto", "inception", "openrouter"}:
    RESEARCH_PLANNER_PROVIDER = "auto"

RESEARCH_SYNTH_PROVIDER = _env_str("RESEARCH_SYNTH_PROVIDER", "auto").strip().lower()
if RESEARCH_SYNTH_PROVIDER not in {"auto", "inception", "openrouter"}:
    RESEARCH_SYNTH_PROVIDER = "auto"

SYNTH_RETRY_ATTEMPTS = max(1, int(_env_str("RESEARCH_SYNTH_RETRY_ATTEMPTS", "2")))
SYNTH_RETRY_DELAY_SECONDS = float(_env_str("RESEARCH_SYNTH_RETRY_DELAY_SECONDS", "0.6"))

BRAVE_API_KEY = _env_str("BRAVE_API_KEY")
BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
BRAVE_DEFAULT_COUNTRY = _env_str("BRAVE_DEFAULT_COUNTRY", "").strip().upper()
BRAVE_DEFAULT_SEARCH_LANG = _env_str("BRAVE_DEFAULT_SEARCH_LANG", "en").strip().lower()
BRAVE_DEFAULT_UI_LANG = _env_str("BRAVE_DEFAULT_UI_LANG", "en-US").strip()
BRAVE_SAFESEARCH = _env_str("BRAVE_SAFESEARCH", "moderate").strip().lower()

TAVILY_API_KEY = _env_str("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TAVILY_MAP_URL = "https://api.tavily.com/map"
TAVILY_CRAWL_URL = "https://api.tavily.com/crawl"
TAVILY_RESEARCH_URL = "https://api.tavily.com/research"
TAVILY_RESEARCH_MODEL = _env_str("TAVILY_RESEARCH_MODEL", "mini")
TAVILY_RESEARCH_POLL_SECONDS = float(_env_str("TAVILY_RESEARCH_POLL_SECONDS", "2.0"))
TAVILY_RESEARCH_TIMEOUT_SECONDS = float(_env_str("TAVILY_RESEARCH_TIMEOUT_SECONDS", "90"))

PLANNER_MAX_TOKENS = 1000  # Increased from 420 to avoid JSON truncation (saw 3037 char responses)
PLANNER_TIMEOUT_SECONDS = 25
SYNTH_MAX_TOKENS = int(_env_str("RESEARCH_SYNTH_MAX_TOKENS", "2600"))
SYNTH_CONTINUATION_MAX_TOKENS = int(_env_str("RESEARCH_SYNTH_CONTINUATION_MAX_TOKENS", "1200"))
SYNTH_MAX_CONTINUATIONS = int(_env_str("RESEARCH_SYNTH_MAX_CONTINUATIONS", "2"))
SYNTH_TIMEOUT_SECONDS = 30

DEFAULT_PROVIDER_MODE = "auto"
DEFAULT_TOOL_MODE = "auto"
DEFAULT_DEPTH = "balanced"

DEPTH_QUERY_LIMIT = {
    "quick": 1,
    "balanced": 3,
    "deep": 5,
}

DEPTH_RESULT_LIMIT = {
    "quick": 5,
    "balanced": 8,
    "deep": 10,
}

MAX_HISTORY_TURNS = 6

BRAVE_MIN_INTERVAL_SECONDS = float(_env_str("BRAVE_MIN_INTERVAL_SECONDS", "1.10"))
BRAVE_RETRY_429_DELAY_SECONDS = float(_env_str("BRAVE_RETRY_429_DELAY_SECONDS", "1.50"))
BRAVE_RETRY_429_MAX_ATTEMPTS = int(_env_str("BRAVE_RETRY_429_MAX_ATTEMPTS", "2"))

BRAVE_MAX_QUERIES_PER_RUN = {
    "quick": int(_env_str("BRAVE_MAX_QUERIES_PER_RUN_QUICK", "1")),
    "balanced": int(_env_str("BRAVE_MAX_QUERIES_PER_RUN_BALANCED", "2")),
    "deep": int(_env_str("BRAVE_MAX_QUERIES_PER_RUN_DEEP", "3")),
}

BRAVE_MAX_REQUESTS_PER_RUN = {
    "quick": int(_env_str("BRAVE_MAX_REQUESTS_PER_RUN_QUICK", "1")),
    "balanced": int(_env_str("BRAVE_MAX_REQUESTS_PER_RUN_BALANCED", "2")),
    "deep": int(_env_str("BRAVE_MAX_REQUESTS_PER_RUN_DEEP", "2")),
}

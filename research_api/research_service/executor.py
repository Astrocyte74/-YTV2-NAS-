"""Deterministic research execution across providers/tools."""

from __future__ import annotations

from collections import OrderedDict
import re
from typing import Callable

from .config import BRAVE_MAX_QUERIES_PER_RUN, BRAVE_MAX_REQUESTS_PER_RUN, DEFAULT_DEPTH, DEPTH_RESULT_LIMIT
from .models import ResearchBatchResult, ResearchItem, ResearchSource
from .planner import plan_research
from .providers.brave import BraveProvider
from .providers.tavily import TavilyProvider, tavily_research_supported
from .providers.base import canonical_domain, extract_domain

_LOW_QUALITY_DOMAIN_SUFFIXES = (
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "quora.com",
    "pinterest.com",
    "linkedin.com",
)
_COMMUNITY_REVIEW_DOMAIN_SUFFIXES = (
    "youtube.com",
    "youtu.be",
    "reddit.com",
)
_MEDIUM_QUALITY_DOMAIN_SUFFIXES = (
    "wikipedia.org",
    "fandom.com",
)
_HIGH_QUALITY_DOMAIN_SUFFIXES = (
    ".gov",
    ".edu",
)
_EXPERT_REVIEW_DOMAIN_HINTS = (
    "caranddriver.com",
    "edmunds.com",
    "motortrend.com",
    "kbb.com",
    "jdpower.com",
    "consumerreports.org",
    "theverge.com",
    "wired.com",
    "cnet.com",
    "techradar.com",
    "pcmag.com",
    "tomsguide.com",
    "rtings.com",
)

ProgressCallback = Callable[[dict], None]

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s)\]>\"']*)?",
    re.IGNORECASE,
)
_SITE_MAP_HINTS = (
    "map ",
    "site structure",
    "identify the pages",
    "find the pages",
    "find pages",
    "relevant pages",
    "which pages",
)
_SITE_CRAWL_HINTS = (
    "crawl ",
    "gather ",
    "collect ",
    "walk the site",
    "traverse ",
    "summarize the site",
    "gather pages",
    "gather content",
    "collect pages",
    "collect content",
)
_SITE_CONTEXT_HINTS = (
    " site",
    " domain",
    " docs",
    " documentation",
    " support",
    " pages",
    " sections",
)
_BROAD_RESEARCH_HINTS = (
    "research",
    "compare",
    "comparison",
    "alternatives",
    "competitors",
    "with sources",
    "what does the web say",
    "what do reviews say",
    "what are people saying",
)
_FRESHNESS_HINTS = (
    "today",
    "latest",
    "current",
    "recent",
    "news",
    "update",
    "updates",
    "release",
    "launch",
    "price",
    "pricing",
    "status",
)


def _emit_progress(progress: ProgressCallback | None, **event: object) -> None:
    if progress is None:
        return
    try:
        progress(event)
    except Exception:
        return


def _looks_like_review_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    review_hints = (
        "review",
        "reviews",
        "owner review",
        "owner reviews",
        "owner feedback",
        "user feedback",
        "what do people say",
        "what are people saying",
        "forum",
        "forums",
        "reddit",
        "youtube review",
        "user sentiment",
    )
    return any(hint in lower for hint in review_hints)


def _domain_quality_score(domain: str, *, allow_community_reviews: bool) -> int:
    clean = canonical_domain(domain)
    if not clean:
        return 0
    if allow_community_reviews and any(clean.endswith(suffix) for suffix in _COMMUNITY_REVIEW_DOMAIN_SUFFIXES):
        return 2
    if any(clean.endswith(suffix) for suffix in _LOW_QUALITY_DOMAIN_SUFFIXES):
        return -3
    if any(clean.endswith(suffix) for suffix in _MEDIUM_QUALITY_DOMAIN_SUFFIXES):
        return 1
    if any(clean.endswith(suffix) for suffix in _HIGH_QUALITY_DOMAIN_SUFFIXES):
        return 5
    if clean.endswith(".org"):
        return 4
    if clean.endswith(".com"):
        return 2
    return 2


def _classify_source_tier(domain: str, *, allow_community_reviews: bool) -> str:
    clean = canonical_domain(domain)
    if not clean:
        return "reference"
    if clean.endswith(".gov") or clean.endswith(".edu") or clean.endswith(".org"):
        return "official"
    if any(clean.endswith(suffix) for suffix in _EXPERT_REVIEW_DOMAIN_HINTS):
        return "expert-review"
    if allow_community_reviews and any(clean.endswith(suffix) for suffix in _COMMUNITY_REVIEW_DOMAIN_SUFFIXES):
        return "community"
    if clean.endswith("wikipedia.org") or clean.endswith("britannica.com"):
        return "reference"
    if clean.endswith(".com"):
        return "expert-review"
    return "reference"


def _item_quality_score(
    *,
    item: ResearchItem,
    provider: str,
    tool: str,
    allow_community_reviews: bool,
) -> int:
    domain = extract_domain(item.url)
    score = _domain_quality_score(domain, allow_community_reviews=allow_community_reviews)
    if tool == "news":
        score += 1
    if tool == "extract":
        score += 1
    if provider == "brave":
        score += 0
    if item.published_at:
        score += 1
    title = (item.title or "").lower()
    if "wikipedia" in title:
        score -= 1
    return score


def _rank_batch_results(batch: ResearchBatchResult, *, allow_community_reviews: bool) -> None:
    batch.results.sort(
        key=lambda item: (
            _item_quality_score(
                item=item,
                provider=batch.provider,
                tool=batch.tool,
                allow_community_reviews=allow_community_reviews,
            ),
            len(item.snippet or ""),
        ),
        reverse=True,
    )


def _filter_low_quality_results(
    batches: list[ResearchBatchResult],
    *,
    allow_community_reviews: bool,
    min_preferred_results: int = 5,
) -> tuple[list[ResearchBatchResult], dict]:
    preferred_total = 0
    low_quality_total = 0
    for batch in batches:
        for item in batch.results:
            domain = extract_domain(item.url)
            if _domain_quality_score(domain, allow_community_reviews=allow_community_reviews) >= 2:
                preferred_total += 1
            else:
                low_quality_total += 1

    if preferred_total < min_preferred_results:
        return batches, {
            "source_quality_filter_applied": False,
            "preferred_result_count": preferred_total,
            "low_quality_result_count": low_quality_total,
            "dropped_low_quality_count": 0,
        }

    filtered_batches: list[ResearchBatchResult] = []
    dropped = 0
    for batch in batches:
        kept_results: list[ResearchItem] = []
        for item in batch.results:
            domain = extract_domain(item.url)
            if _domain_quality_score(domain, allow_community_reviews=allow_community_reviews) >= 2:
                kept_results.append(item)
            else:
                dropped += 1

        if kept_results:
            filtered_batches.append(
                ResearchBatchResult(
                    query=batch.query,
                    provider=batch.provider,
                    tool=batch.tool,
                    results=kept_results,
                    errors=list(batch.errors),
                    latency_ms=batch.latency_ms,
                    rate_limited=batch.rate_limited,
                    retry_count=batch.retry_count,
                )
            )

    return filtered_batches, {
        "source_quality_filter_applied": dropped > 0,
        "preferred_result_count": preferred_total,
        "low_quality_result_count": low_quality_total,
        "dropped_low_quality_count": dropped,
    }


def _query_prefers_tavily_provider(query: str) -> bool:
    shape = _analyze_query_shape(query)
    return bool(shape["has_url"]) or (
        bool(shape["has_domain"])
        and (bool(shape["has_site_context"]) or bool(shape["wants_map"]) or bool(shape["wants_crawl"]))
    )


def _extract_site_target(text: str) -> str:
    url_match = _URL_RE.search(text or "")
    if url_match:
        return url_match.group(0).rstrip(".,;")
    domain_match = _DOMAIN_RE.search(text or "")
    if domain_match:
        return domain_match.group(0).rstrip(".,;")
    normalized = " ".join((text or "").replace("’", "'").split())
    support_match = re.search(r"\b([a-z0-9][a-z0-9-]{1,62})('?s)?\s+(support|official)\s+site\b", normalized, re.IGNORECASE)
    if support_match:
        return f"{support_match.group(1).lower()}.com"
    return ""


def _analyze_query_shape(text: str) -> dict[str, bool]:
    lower_query = f" {(text or '').strip().lower()} "
    return {
        "has_url": bool(_URL_RE.search(text or "")),
        "has_domain": bool(_DOMAIN_RE.search(text or "")),
        "has_site_context": any(hint in lower_query for hint in _SITE_CONTEXT_HINTS),
        "wants_map": any(hint in lower_query for hint in _SITE_MAP_HINTS),
        "wants_crawl": any(hint in lower_query for hint in _SITE_CRAWL_HINTS),
        "wants_broad_research": any(hint in lower_query for hint in _BROAD_RESEARCH_HINTS),
    }


def _looks_fresh_query(text: str) -> bool:
    lower_query = f" {(text or '').strip().lower()} "
    return any(hint in lower_query for hint in _FRESHNESS_HINTS)


def _ensure_site_target(query: str, message: str) -> str:
    if _extract_site_target(query):
        return query
    site_target = _extract_site_target(message)
    if not site_target:
        return query
    return f"{site_target} {query}".strip()


def _resolve_providers(
    provider_mode: str,
    compare: bool,
    queries: list[str] | None = None,
    *,
    tool_mode: str = "auto",
    requires_source_backed_answer: bool = False,
) -> list[str]:
    if compare or provider_mode == "both":
        return ["brave", "tavily"]
    if provider_mode in {"brave", "tavily"}:
        return [provider_mode]
    if any(_query_prefers_tavily_provider(query) for query in (queries or [])):
        return ["tavily"]
    if tool_mode == "deep" and requires_source_backed_answer:
        return ["tavily"]
    # Auto mode uses Brave first; Tavily is used reactively as fallback.
    return ["brave"]


def _resolve_tools(
    provider: str,
    query: str,
    message: str,
    tool_mode: str,
    manual_tools: dict | None,
    *,
    freshness_sensitive: bool,
    requires_source_backed_answer: bool,
    compare: bool,
) -> tuple[list[str], str]:
    manual_tools = manual_tools or {}
    if tool_mode == "manual":
        selected = manual_tools.get(provider) or []
        out = [str(x).strip().lower() for x in selected if str(x).strip()]
        tools = out or (["web"] if provider == "brave" else ["search"])
        return tools, "manual tool selection"

    if provider == "brave":
        if freshness_sensitive or _looks_fresh_query(query) or _looks_fresh_query(message):
            return ["news", "web"], "freshness-sensitive request prefers Brave news first"
        if tool_mode in {"deep"}:
            return ["web"], "deep mode keeps Brave on web to avoid rate-limit fan-out"
        return ["web"], "default Brave web search"

    query_shape = _analyze_query_shape(query)
    message_shape = _analyze_query_shape(message)
    has_site_target = bool(_extract_site_target(query) or _extract_site_target(message))

    if has_site_target and (bool(query_shape["wants_crawl"]) or bool(message_shape["wants_crawl"])):
        return ["crawl"], "explicit site/domain request to gather content across pages"
    if has_site_target and (
        bool(query_shape["wants_map"])
        or bool(query_shape["has_site_context"])
        or bool(message_shape["wants_map"])
    ):
        return ["map"], "explicit site/domain request to discover relevant pages"
    if bool(query_shape["has_url"]):
        return ["extract"], "explicit URL request favors extraction"

    if tool_mode == "deep":
        if (
            tavily_research_supported()
            and (requires_source_backed_answer or compare or bool(query_shape["wants_broad_research"]) or bool(message_shape["wants_broad_research"]))
        ):
            return ["research"], "deep mode broad source-backed synthesis"
        return ["search", "extract"], "deep mode richer Tavily retrieval"

    if tool_mode == "auto" and (requires_source_backed_answer or compare):
        return ["search", "extract"], "auto mode source-backed answer path"
    return ["search"], "default Tavily search"


def _limit_queries_for_provider(provider: str, queries: list[str], depth: str, tool_count: int = 1) -> list[str]:
    if provider != "brave":
        return queries
    max_queries = BRAVE_MAX_QUERIES_PER_RUN.get(depth, BRAVE_MAX_QUERIES_PER_RUN.get(DEFAULT_DEPTH, 2))
    max_queries = max(1, int(max_queries))
    request_budget = BRAVE_MAX_REQUESTS_PER_RUN.get(depth, BRAVE_MAX_REQUESTS_PER_RUN.get(DEFAULT_DEPTH, max_queries))
    request_budget = max(1, int(request_budget))
    tool_count = max(1, int(tool_count))
    max_queries = min(max_queries, max(1, request_budget // tool_count))
    return queries[:max_queries]


def _build_provider_options(
    *,
    provider: str,
    tool: str,
    query: str,
    message: str,
    depth: str,
    freshness_sensitive: bool,
) -> dict:
    if provider != "brave":
        return {}
    return {
        "depth": depth,
        "freshness_sensitive": bool(freshness_sensitive or _looks_fresh_query(query) or _looks_fresh_query(message)),
        "freshness": "pw" if tool == "news" else None,
        "extra_snippets": depth in {"balanced", "deep"},
        "query_source": "message" if query == message else "planned_query",
    }


def execute_research(
    *,
    message: str,
    history: list[dict[str, str]] | None,
    provider_mode: str,
    tool_mode: str,
    depth: str,
    compare: bool,
    manual_tools: dict | None,
    progress: ProgressCallback | None = None,
) -> tuple[list[ResearchBatchResult], list[ResearchSource], dict, str | None]:
    """Run planned research and return normalized results + sources + meta.

    Returns:
      (batches, deduped_sources, meta, clarification_question)
    """
    clean_depth = depth if depth in {"quick", "balanced", "deep"} else DEFAULT_DEPTH

    _emit_progress(
        progress,
        type="progress",
        stage="planning_started",
        label="Planning research...",
        message=message,
    )

    plan = plan_research(
        message=message,
        history=history,
        provider_mode=provider_mode,
        tool_mode=tool_mode,
        depth=clean_depth,
    )

    _emit_progress(
        progress,
        type="progress",
        stage="planning_completed",
        label=f"Prepared {len(plan.queries)} quer{'y' if len(plan.queries) == 1 else 'ies'}",
        objective=plan.objective,
        queries=plan.queries,
        provider_mode=plan.provider_mode,
        tool_mode=plan.tool_mode,
        needs_clarification=plan.needs_clarification,
    )

    if plan.needs_clarification and plan.clarification_question:
        return [], [], {
            "status": "clarify",
            "objective": plan.objective,
            "provider_mode": plan.provider_mode,
            "tool_mode": plan.tool_mode,
            "planner_llm_provider": plan.llm_provider or "unknown",
            "planner_llm_model": plan.llm_model or "unknown",
            "queries": plan.queries,
            "entities": plan.entities,
            "comparison_axes": plan.comparison_axes,
            "freshness_sensitive": plan.freshness_sensitive,
            "requires_source_backed_answer": plan.requires_source_backed_answer,
        }, plan.clarification_question

    providers = _resolve_providers(
        plan.provider_mode,
        compare,
        plan.queries,
        tool_mode=plan.tool_mode,
        requires_source_backed_answer=plan.requires_source_backed_answer,
    )
    result_limit = DEPTH_RESULT_LIMIT.get(clean_depth, DEPTH_RESULT_LIMIT[DEFAULT_DEPTH])
    allow_community_reviews = _looks_like_review_request(message)

    provider_clients = {
        "brave": BraveProvider(),
        "tavily": TavilyProvider(),
    }

    batches: list[ResearchBatchResult] = []
    provider_errors: list[str] = []
    rate_limit_events: list[dict] = []
    fallback_events: list[dict] = []
    tool_decisions: list[dict] = []

    for provider in providers:
        tool_count_hint = 1
        if provider == "brave" and plan.queries:
            hinted_tools, _ = _resolve_tools(
                provider,
                plan.queries[0],
                message,
                plan.tool_mode,
                manual_tools,
                freshness_sensitive=plan.freshness_sensitive,
                requires_source_backed_answer=plan.requires_source_backed_answer,
                compare=compare,
            )
            tool_count_hint = len(hinted_tools) or 1
        provider_queries = _limit_queries_for_provider(provider, plan.queries, clean_depth, tool_count_hint)
        if provider == "tavily" and plan.tool_mode != "manual" and _URL_RE.search(message or ""):
            provider_queries = [message]
        for query in provider_queries:
            client = provider_clients.get(provider)
            if client is None:
                continue
            tools, tool_reason = _resolve_tools(
                provider,
                query,
                message,
                plan.tool_mode,
                manual_tools,
                freshness_sensitive=plan.freshness_sensitive,
                requires_source_backed_answer=plan.requires_source_backed_answer,
                compare=compare,
            )
            for tool in tools:
                provider_options = _build_provider_options(
                    provider=provider,
                    tool=tool,
                    query=query,
                    message=message,
                    depth=clean_depth,
                    freshness_sensitive=plan.freshness_sensitive,
                )
                tool_decisions.append({
                    "provider": provider,
                    "query": query,
                    "tools": [tool],
                    "reason": tool_reason,
                    "site_target": _extract_site_target(query) or _extract_site_target(message) or None,
                    "options": provider_options or None,
                })
                effective_query = (
                    _ensure_site_target(query, message)
                    if provider == "tavily" and tool in {"map", "crawl"}
                    else query
                )
                verb = "Checking"
                if tool == "extract":
                    verb = "Extracting"
                elif tool == "crawl":
                    verb = "Crawling"
                elif tool == "map":
                    verb = "Mapping"
                elif tool == "research":
                    verb = "Researching"
                elif tool == "news":
                    verb = "Checking news"
                _emit_progress(
                    progress,
                    type="progress",
                    stage="provider_run_started",
                    label=f"{verb} {provider} {tool}: {query}",
                    provider=provider,
                    tool=tool,
                    query=query,
                    reason=tool_reason,
                )
                batch = client.execute(query=effective_query, tool=tool, max_results=result_limit, options=provider_options)
                if batch.errors:
                    provider_errors.extend(batch.errors)
                if (
                    provider == "tavily"
                    and tool == "research"
                    and not batch.results
                    and any(
                        marker in str(err).lower()
                        for err in batch.errors
                        for marker in (
                            "432",
                            "set usage limit",
                            "research exceeded the current plan limit",
                        )
                    )
                ):
                    _emit_progress(
                        progress,
                        type="progress",
                        stage="fallback_triggered",
                        label=f"Tavily research unavailable; falling back to search/extract for: {query}",
                        from_provider="tavily",
                        to_provider="tavily",
                        reason="research_432",
                        query=query,
                    )
                    fallback_events.append({
                        "from": "tavily",
                        "to": "tavily",
                        "reason": "research_432",
                        "query": query,
                    })
                    fallback_tools = ["search", "extract"]
                    for fallback_tool in fallback_tools:
                        fallback_batch = client.execute(query=query, tool=fallback_tool, max_results=result_limit, options={})
                        if fallback_batch.errors:
                            provider_errors.extend(fallback_batch.errors)
                        if fallback_batch.results:
                            batches.append(fallback_batch)
                    continue
                if batch.results:
                    batches.append(batch)
                top_domains: list[str] = []
                seen_domains: set[str] = set()
                for item in batch.results:
                    domain = extract_domain(item.url)
                    if not domain or domain in seen_domains:
                        continue
                    seen_domains.add(domain)
                    top_domains.append(domain)
                    if len(top_domains) >= 3:
                        break
                _emit_progress(
                    progress,
                    type="progress",
                    stage="provider_run_finished",
                    label=(
                        f"{provider} {tool}: {len(batch.results)} result{'s' if len(batch.results) != 1 else ''}"
                        + (f" from {', '.join(top_domains)}" if top_domains else "")
                    ),
                    provider=provider,
                    tool=tool,
                    query=query,
                    result_count=len(batch.results),
                    latency_ms=batch.latency_ms,
                    domains=top_domains,
                    errors=batch.errors[:3],
                )
                if batch.rate_limited:
                    rate_limit_events.append({
                        "provider": provider,
                        "tool": tool,
                        "query": query,
                        "retry_count": batch.retry_count,
                    })
                    # Only fall back if Brave stayed empty after retrying. A recovered
                    # Brave response should not duplicate work or deflect credit to Tavily.
                    tavily = provider_clients.get("tavily")
                    if tavily and not batch.results:
                        _emit_progress(
                            progress,
                            type="progress",
                            stage="fallback_triggered",
                            label=f"Falling back from brave to tavily for: {query}",
                            from_provider="brave",
                            to_provider="tavily",
                            reason="rate_limited",
                            query=query,
                        )
                        tavily_batch = tavily.execute(query=query, tool="search", max_results=result_limit, options={})
                        if tavily_batch.results:
                            batches.append(tavily_batch)
                            fallback_events.append({
                                "from": "brave",
                                "to": "tavily",
                                "reason": "rate_limited",
                                "query": query,
                            })
                        if tavily_batch.errors:
                            provider_errors.extend(tavily_batch.errors)

    # Auto fallback if no data returned from Brave-only path.
    if not batches and providers == ["brave"]:
        tavily = provider_clients["tavily"]
        for query in plan.queries:
            _emit_progress(
                progress,
                type="progress",
                stage="fallback_triggered",
                label=f"No brave results; trying tavily for: {query}",
                from_provider="brave",
                to_provider="tavily",
                reason="empty_results",
                query=query,
            )
            fallback = tavily.execute(query=query, tool="search", max_results=result_limit, options={})
            if fallback.results:
                batches.append(fallback)
                fallback_events.append({
                    "from": "brave",
                    "to": "tavily",
                    "reason": "empty_results",
                    "query": query,
                })
            if fallback.errors:
                provider_errors.extend(fallback.errors)

    for batch in batches:
        _rank_batch_results(batch, allow_community_reviews=allow_community_reviews)

    batches, quality_meta = _filter_low_quality_results(
        batches,
        allow_community_reviews=allow_community_reviews,
    )

    source_map: OrderedDict[str, ResearchSource] = OrderedDict()
    provider_source_urls: dict[str, set[str]] = {}
    for batch in batches:
        provider_urls = provider_source_urls.setdefault(batch.provider, set())
        for item in batch.results:
            url = item.url.strip()
            if not url:
                continue
            provider_urls.add(url)
            existing = source_map.get(url)
            if existing is None:
                domain = extract_domain(url)
                source_map[url] = ResearchSource(
                    name=item.title.strip() or extract_domain(url),
                    url=url,
                    domain=domain,
                    tier=_classify_source_tier(domain, allow_community_reviews=allow_community_reviews),
                    providers=[batch.provider],
                    tools=[batch.tool],
                )
                continue

            # Track all providers/tools that surfaced this source URL.
            if batch.provider and batch.provider not in existing.providers:
                existing.providers.append(batch.provider)
            if batch.tool and batch.tool not in existing.tools:
                existing.tools.append(batch.tool)

    sources = list(source_map.values())

    batch_meta = [
        {
            "provider": b.provider,
            "tool": b.tool,
            "query": b.query,
            "latency_ms": b.latency_ms,
            "result_count": len(b.results),
            "error_count": len(b.errors),
        }
        for b in batches
    ]

    by_provider: dict[str, dict] = {}
    for row in batch_meta:
        provider = row["provider"]
        entry = by_provider.setdefault(provider, {
            "provider": provider,
            "tool_runs": 0,
            "tools": set(),
            "result_count": 0,
            "latency_total_ms": 0,
            "queries": set(),
        })
        entry["tool_runs"] += 1
        entry["tools"].add(row["tool"])
        entry["queries"].add(row["query"])
        entry["result_count"] += int(row["result_count"])
        entry["latency_total_ms"] += int(row["latency_ms"])

    provider_summary = []
    for provider, row in by_provider.items():
        tool_runs = max(1, int(row["tool_runs"]))
        provider_summary.append({
            "provider": provider,
            "tool_runs": tool_runs,
            "tools": sorted(list(row["tools"])),
            "query_count": len(row["queries"]),
            "result_count": int(row["result_count"]),
            "avg_latency_ms": int(row["latency_total_ms"] / tool_runs),
            "unique_source_count": len(provider_source_urls.get(provider, set())),
        })

    meta = {
        "status": "ok" if batches else "fallback",
        "objective": plan.objective,
        "entities": plan.entities,
        "comparison_axes": plan.comparison_axes,
        "freshness_sensitive": bool(plan.freshness_sensitive),
        "requires_source_backed_answer": bool(plan.requires_source_backed_answer),
        "planner_llm_provider": plan.llm_provider or "unknown",
        "planner_llm_model": plan.llm_model or "unknown",
        "allow_community_review_sources": bool(allow_community_reviews),
        "provider_mode": plan.provider_mode,
        "tool_mode": plan.tool_mode,
        "queries": plan.queries,
        "provider_chain": providers,
        "batch_count": len(batches),
        "result_count": sum(len(b.results) for b in batches),
        "source_count": len(sources),
        "source_domains": [src.domain for src in sources[:12] if src.domain],
        "errors": provider_errors[:10],
        "compare": bool(compare),
        "depth": clean_depth,
        "batches": batch_meta,
        "by_provider": provider_summary,
        "rate_limit_events": rate_limit_events,
        "fallback_events": fallback_events,
        "tool_decisions": tool_decisions,
        **quality_meta,
    }

    if rate_limit_events:
        meta["user_notice"] = (
            "Brave hit a temporary rate limit on some queries. "
            "Fallback providers were used where available."
        )
    elif any(
        decision.get("provider") == "tavily"
        and any(tool in {"extract", "map", "crawl", "research"} for tool in decision.get("tools", []))
        for decision in tool_decisions
    ):
        meta["user_notice"] = (
            "Auto strategy used Tavily's task-specific tools for this query shape "
            "(for example URL extraction, site mapping/crawling, or deep research)."
        )
    elif quality_meta.get("source_quality_filter_applied"):
        meta["user_notice"] = (
            "Lower-quality social, forum, and video sources were filtered out because stronger sources were available."
        )

    return batches, sources, meta, None

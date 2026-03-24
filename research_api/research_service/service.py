"""Public entry points for portable research execution."""

from __future__ import annotations

import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Callable

from .config import (
    BRAVE_API_KEY,
    INCEPTION_API_KEY,
    INCEPTION_MODEL,
    OPENROUTER_API_KEY,
    RESEARCH_ENABLED,
    RESEARCH_FALLBACK_ENABLED,
    RESEARCH_FALLBACK_MODEL,
    RESEARCH_PLANNER_PROVIDER,
    RESEARCH_SYNTH_PROVIDER,
    TAVILY_API_KEY,
)
from .executor import execute_research, execute_research_plan
from .follow_up import (
    FollowUpResearchPlan,
    FollowUpSuggestion,
    MAX_APPROVED_QUESTIONS,
    build_cache_key,
    plan_follow_up_research,
    suggest_follow_up_questions,
)
from .models import ResearchRunResult
from .providers.tavily import tavily_research_supported
from .synthesizer import answer_report_chat, synthesize_answer, synthesize_follow_up

logger = logging.getLogger(__name__)

_FOLLOW_UP_RESEARCH_CACHE: dict[str, ResearchRunResult] = {}


def clear_follow_up_research_cache() -> None:
    """Reset the in-process follow-up research cache."""
    _FOLLOW_UP_RESEARCH_CACHE.clear()


def get_cached_follow_up_research(cache_key: str) -> ResearchRunResult | None:
    """Return a defensive copy of a cached follow-up research result."""
    cached = _FOLLOW_UP_RESEARCH_CACHE.get(cache_key)
    if cached is None:
        return None
    return deepcopy(cached)


def store_follow_up_research_result(cache_key: str, result: ResearchRunResult) -> None:
    """Store a defensive copy of a follow-up research result in cache."""
    _FOLLOW_UP_RESEARCH_CACHE[cache_key] = deepcopy(result)


def get_research_capabilities() -> dict:
    tavily_tools = ["search", "extract", "crawl", "map"]
    if tavily_research_supported():
        tavily_tools.append("research")
    return {
        "enabled": RESEARCH_ENABLED,
        "llm_configured": bool(INCEPTION_API_KEY) or bool(OPENROUTER_API_KEY),
        "llm_primary": "inception" if INCEPTION_API_KEY else "openrouter",
        "llm_primary_model": INCEPTION_MODEL if INCEPTION_API_KEY else RESEARCH_FALLBACK_MODEL,
        "llm_fallback": {
            "enabled": RESEARCH_FALLBACK_ENABLED,
            "available": bool(OPENROUTER_API_KEY),
            "model": RESEARCH_FALLBACK_MODEL,
        },
        "llm_stage_overrides": {
            "planner": RESEARCH_PLANNER_PROVIDER,
            "synth": RESEARCH_SYNTH_PROVIDER,
        },
        "providers": {
            "brave": {
                "enabled": bool(BRAVE_API_KEY),
                "tools": ["web", "news"],
            },
            "tavily": {
                "enabled": bool(TAVILY_API_KEY),
                "tools": tavily_tools,
            },
        },
        "defaults": {
            "provider_mode": "auto",
            "tool_mode": "auto",
            "depth": "balanced",
            "compare": False,
        },
    }


def run_research(
    *,
    message: str,
    history: list[dict[str, str]] | None,
    provider_mode: str,
    tool_mode: str,
    depth: str,
    compare: bool,
    manual_tools: dict | None,
    progress: Callable[[dict], None] | None = None,
) -> ResearchRunResult:
    if not RESEARCH_ENABLED:
        return ResearchRunResult(
            answer="Research mode is currently disabled.",
            sources=[],
            status="error",
            meta={"status": "error", "error": "disabled"},
        )

    batches, sources, meta, clarification_question = execute_research(
        message=message,
        history=history,
        provider_mode=provider_mode,
        tool_mode=tool_mode,
        depth=depth,
        compare=compare,
        manual_tools=manual_tools,
        progress=progress,
    )

    if clarification_question:
        return ResearchRunResult(
            answer=clarification_question,
            sources=[],
            status="fallback",
            meta={**meta, "status": "clarify", "clarification_question": clarification_question},
        )

    if progress is not None:
        try:
            progress({
                "type": "progress",
                "stage": "synthesis_started",
                "label": "Writing answer...",
            })
        except Exception:
            pass

    answer, synth_llm_info = synthesize_answer(
        user_message=message,
        history=history,
        batches=batches,
        sources=sources,
        compare=compare,
    )

    meta["llm_provider"] = synth_llm_info.get("llm_provider", "unknown")
    meta["llm_model"] = synth_llm_info.get("llm_model", "unknown")
    meta["synth_llm_provider"] = synth_llm_info.get("llm_provider", "unknown")
    meta["synth_llm_model"] = synth_llm_info.get("llm_model", "unknown")
    logger.info("Research completed: provider=%s model=%s", meta.get("llm_provider"), meta.get("llm_model"))

    return ResearchRunResult(
        answer=answer,
        sources=sources,
        status="ok" if batches else "fallback",
        meta=meta,
    )


def serialize_research_run(run: ResearchRunResult) -> dict[str, object]:
    return {
        "status": run.status,
        "response": run.answer,
        "sources": [
            {
                "name": source.name,
                "url": source.url,
                "domain": source.domain,
                "tier": source.tier,
                "providers": source.providers,
                "tools": source.tools,
            }
            for source in run.sources
        ],
        "meta": run.meta,
        "error": None if run.status != "error" else run.meta.get("error"),
    }


def get_follow_up_suggestions(
    *,
    source_context: dict,
    summary: str,
    entities: list[str] | None = None,
    max_suggestions: int = 3,
) -> list[dict]:
    """Generate follow-up research suggestions for a summary.

    Args:
        source_context: Information about the original content (url, type, title, etc.)
        summary: The existing summary text
        entities: Optional list of extracted entities
        max_suggestions: Maximum number of suggestions to generate

    Returns:
        List of suggestion dicts with keys: id, label, question, reason, kind, priority, default_selected
    """
    if not RESEARCH_ENABLED:
        return []

    suggestions = suggest_follow_up_questions(
        source_context=source_context,
        summary=summary,
        entities=entities,
        max_suggestions=max_suggestions,
    )

    return [
        {
            "id": s.id,
            "label": s.label,
            "question": s.question,
            "reason": s.reason,
            "kind": s.kind,
            "priority": s.priority,
            "default_selected": s.default_selected,
            "provenance": s.provenance,
        }
        for s in suggestions
    ]


def run_follow_up_research(
    *,
    source_context: dict,
    summary: str,
    approved_questions: list[str],
    question_provenance: list[str] | None = None,
    summary_id: int | None = None,
    provider_mode: str = "auto",
    tool_mode: str = "auto",
    depth: str = "balanced",
    compare: bool = False,
    manual_tools: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> ResearchRunResult:
    """Run follow-up research based on approved user questions.

    This is the main entry point for follow-up research. It:
    1. Takes approved user questions
    2. Consolidates them into a minimal research plan
    3. Executes the consolidated plan
    4. Synthesizes a sectioned report answering each question

    Args:
        source_context: Information about the original content (url, type, title, etc.)
        summary: The existing summary text
        approved_questions: User-approved research directions (max 3)
        question_provenance: Where each question came from (suggested/preset/custom)
        summary_id: Optional specific summary revision ID for cache invalidation.
            If None, cache key uses "latest" which may not distinguish between
            summary revisions.
        provider_mode: Research provider mode
        tool_mode: Research tool mode
        depth: Research depth
        compare: DEPRECATED: Comparison intent is now inferred from questions
            by the planner. This parameter is ignored in favor of plan.compare.
        manual_tools: Optional manual tool selection per provider
        progress: Optional progress callback

    Returns:
        ResearchRunResult with sectioned report answering each approved question
    """
    if not RESEARCH_ENABLED:
        return ResearchRunResult(
            answer="Research mode is currently disabled.",
            sources=[],
            status="error",
            meta={"status": "error", "error": "disabled"},
        )

    if not approved_questions:
        return ResearchRunResult(
            answer="No approved questions provided for follow-up research.",
            sources=[],
            status="error",
            meta={"status": "error", "error": "no_questions"},
        )

    if len(approved_questions) > MAX_APPROVED_QUESTIONS:
        return ResearchRunResult(
            answer=f"Maximum {MAX_APPROVED_QUESTIONS} questions allowed for follow-up research.",
            sources=[],
            status="error",
            meta={"status": "error", "error": "too_many_questions"},
        )

    cache_key = build_cache_key(
        video_id=source_context.get("video_id", source_context.get("id", "")),
        summary_id=summary_id,
        approved_questions=approved_questions,
        provider_mode=provider_mode,
        depth=depth,
    )
    cached_result = get_cached_follow_up_research(cache_key)
    if cached_result is not None:
        cached_result.meta["cache_key"] = cache_key
        cached_result.meta["cache_hit"] = True
        if progress is not None:
            try:
                progress({
                    "type": "progress",
                    "stage": "followup_cache_hit",
                    "label": "Using cached follow-up research result.",
                    "cache_key": cache_key,
                })
            except Exception:
                pass
        return cached_result

    # Step 1: Generate consolidated research plan
    if progress is not None:
        try:
            progress({
                "type": "progress",
                "stage": "followup_planning_started",
                "label": "Planning follow-up research...",
                "approved_questions": approved_questions,
            })
        except Exception:
            pass

    plan = plan_follow_up_research(
        source_context=source_context,
        summary=summary,
        approved_questions=approved_questions,
        question_provenance=question_provenance,
        provider_mode=provider_mode,
        tool_mode=tool_mode,
        depth=depth,
    )

    # Step 2: Execute the consolidated plan
    if progress is not None:
        try:
            progress({
                "type": "progress",
                "stage": "followup_planning_completed",
                "label": f"Prepared {len(plan.planned_queries)} consolidated quer{'y' if len(plan.planned_queries) == 1 else 'ies'} for {len(approved_questions)} question{'s' if len(approved_questions) > 1 else ''}",
                "planned_queries": plan.planned_queries,
                "approved_questions": approved_questions,
                "coverage_map": plan.coverage_map,
                "dedupe_notes": plan.dedupe_notes,
            })
        except Exception:
            pass

    message = "; ".join(approved_questions)
    batches, sources, meta = execute_research_plan(
        plan=plan,
        message=message,
        depth=depth,
        compare=plan.compare,  # Use planner-inferred comparison intent
        manual_tools=manual_tools,
        progress=progress,
    )

    # Step 3: Synthesize sectioned report
    if progress is not None:
        try:
            progress({
                "type": "progress",
                "stage": "synthesis_started",
                "label": "Writing follow-up report...",
            })
        except Exception:
            pass

    answer, synth_llm_info = synthesize_follow_up(
        source_context=source_context,
        summary=summary,
        approved_questions=approved_questions,
        question_kinds=plan.question_kinds,
        batches=batches,
        sources=sources,
        compare=plan.compare,  # Use planner-inferred comparison intent
    )

    meta["synth_llm_provider"] = synth_llm_info.get("llm_provider", "unknown")
    meta["synth_llm_model"] = synth_llm_info.get("llm_model", "unknown")
    meta["cache_key"] = cache_key
    meta["cache_hit"] = False

    logger.info(
        "Follow-up research completed: provider=%s model=%s questions=%d",
        meta.get("synth_llm_provider"),
        meta.get("synth_llm_model"),
        len(approved_questions),
    )

    result = ResearchRunResult(
        answer=answer,
        sources=sources,
        status="ok" if batches else "fallback",
        meta=meta,
    )
    store_follow_up_research_result(cache_key, result)
    return result


def answer_follow_up_chat(
    *,
    source_context: dict,
    report_answer: str,
    report_sources: list[dict] | None,
    user_question: str,
    history: list[dict[str, str]] | None = None,
    thread_turns: list[dict] | None = None,
) -> tuple[str, dict[str, str], list[dict]]:
    """Answer a lightweight follow-up question using an existing report only."""
    normalized_sources = []
    for source in report_sources or []:
        if hasattr(source, "url"):
            normalized_sources.append(source)
            continue
        if not isinstance(source, dict):
            continue
        normalized_sources.append(SimpleNamespace(
            name=str(source.get("name") or ""),
            url=str(source.get("url") or ""),
            domain=str(source.get("domain") or ""),
            tier=str(source.get("tier") or ""),
            providers=list(source.get("providers") or []),
            tools=list(source.get("tools") or []),
        ))

    answer, llm_info = answer_report_chat(
        source_context=source_context,
        report_answer=report_answer,
        report_sources=normalized_sources,
        user_question=user_question,
        history=history,
        thread_turns=thread_turns,
    )
    serialized_sources = [
        {
            "name": getattr(source, "name", ""),
            "url": getattr(source, "url", ""),
            "domain": getattr(source, "domain", ""),
            "tier": getattr(source, "tier", ""),
            "providers": list(getattr(source, "providers", []) or []),
            "tools": list(getattr(source, "tools", []) or []),
        }
        for source in normalized_sources
    ]
    return answer, llm_info, serialized_sources

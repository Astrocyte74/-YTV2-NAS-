"""Public entry points for portable research execution."""

from __future__ import annotations

import logging
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
from .executor import execute_research
from .models import ResearchRunResult
from .providers.tavily import tavily_research_supported
from .synthesizer import synthesize_answer

logger = logging.getLogger(__name__)


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

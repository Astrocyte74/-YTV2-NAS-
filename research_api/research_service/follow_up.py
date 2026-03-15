"""Follow-up research orchestration for YTV2.

This module provides the core logic for generating follow-up research
suggestions and executing coordinated follow-up research runs.

Key design principles:
- Multiple approved questions → ONE coordinated research plan
- Planner consolidates and dedupes before execution
- Single shared source set across all questions
- Clear separation between user-facing questions and internal search queries
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from .config import (
    DEFAULT_DEPTH,
    DEFAULT_PROVIDER_MODE,
    DEFAULT_TOOL_MODE,
    PLANNER_MAX_TOKENS,
    PLANNER_TIMEOUT_SECONDS,
    RESEARCH_PLANNER_PROVIDER,
)
from .llm import chat_json_schema
from .models import ResearchPlan

logger = logging.getLogger(__name__)

# Constants
MAX_APPROVED_QUESTIONS = 3
MAX_PLANNED_QUERIES = 6

# Type definitions
QuestionKind = Literal[
    "current_state",
    "pricing",
    "comparison",
    "alternatives",
    "fact_check",
    "background",
    "what_changed",
]

QuestionProvenance = Literal["suggested", "preset", "custom"]


# JSON Schema for LLM-generated suggestions
SUGGESTIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["current_state", "pricing", "comparison", "alternatives", "fact_check", "background", "what_changed"],
                    },
                    "priority": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "default_selected": {"type": "boolean"},
                },
                "required": ["id", "label", "question", "reason", "kind", "priority", "default_selected"],
            },
            "minItems": 1,
            "maxItems": 3,
        },
        "should_suggest": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["suggestions", "should_suggest", "explanation"],
}


SUGGESTIONS_SYSTEM_PROMPT = """You are a follow-up research suggestion engine.

Your job is to analyze a summary and determine what follow-up research questions would be most valuable.

**When to suggest:**
- Content is time-sensitive (older than 3-6 months, mentions pricing/products/releases)
- Content discusses comparisons or alternatives
- Content may be outdated or have recent updates
- Content mentions facts that could be verified

**When NOT to suggest:**
- Evergreen educational content with no changing factual surface
- Purely personal/opinion content
- Very recent content (< 1 week) unless pricing/comparisons are mentioned
- Content that is already current and comprehensive

**Your output must include:**
1. suggestions: 1-3 contextual follow-up questions
2. should_suggest: whether this content warrants follow-up research at all
3. explanation: brief reasoning for your suggestion decision

**Suggestion quality guidelines:**
- Questions should be specific and actionable
- Questions should reflect what users would actually want to know
- Questions should be answerable via web research
- Prioritize questions that provide high value (pricing, current state, comparisons)

**Output format:**
{{
  "suggestions": [
    {{
      "id": "short-unique-id",
      "label": "Short user-facing label",
      "question": "Specific question the user wants answered",
      "reason": "Why this question matters for this content",
      "kind": "pricing|comparison|what_changed|current_state|alternatives|fact_check|background",
      "priority": 0.9,
      "default_selected": true
    }}
  ],
  "should_suggest": true,
  "explanation": "Content discusses X which changes frequently; users would want to know Y"
}}

**Kind definitions:**
- pricing: Current pricing, costs, tiers, availability
- comparison: How things compare to alternatives
- what_changed: What changed since publication
- current_state: Current status or state of something
- alternatives: What alternatives exist
- fact_check: Verify specific claims
- background: Additional context or background info
"""


@dataclass
class FollowUpSuggestion:
    """A suggested follow-up research question."""

    id: str
    label: str
    question: str
    reason: str
    kind: QuestionKind
    priority: float  # 0.0 to 1.0
    default_selected: bool
    provenance: QuestionProvenance = "suggested"


@dataclass
class FollowUpResearchPlan:
    """Consolidated research plan for follow-up research."""

    approved_questions: list[str]
    question_provenance: list[QuestionProvenance]
    question_kinds: list[QuestionKind]
    planned_queries: list[str]
    coverage_map: list[dict]  # Maps approved questions to planned queries
    dedupe_notes: str

    # Plan metadata
    provider_mode: str
    tool_mode: str
    depth: str
    compare: bool = False
    freshness_sensitive: bool = False

    # LLM info
    planner_provider: str = ""
    planner_model: str = ""


def _build_cache_key(
    video_id: str,
    summary_id: Optional[int],
    approved_questions: list[str],
    provider_mode: str,
    depth: str,
) -> str:
    """
    Build cache key for follow-up research results.

    Includes:
    - video_id: content identity
    - summary_id: specific summary revision (if available)
    - normalized_questions: sorted, deduped approved questions
    - provider_mode: research provider strategy
    - depth: research depth setting
    """
    # Normalize questions: sort and dedupe
    normalized_questions = sorted(set(q.strip() for q in approved_questions if q.strip()))
    questions_hash = hashlib.md5("|".join(normalized_questions).encode()).hexdigest()[:8]

    parts = [
        video_id,
        str(summary_id) if summary_id else "latest",
        questions_hash,
        provider_mode,
        depth,
    ]

    return ":".join(parts)


FOLLOW_UP_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "planned_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_PLANNED_QUERIES,
        },
        "coverage_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "approved_question": {"type": "string"},
                    "covered_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["approved_question", "covered_by"],
            },
            "minItems": 1,
            "maxItems": MAX_APPROVED_QUESTIONS,
        },
        "dedupe_notes": {"type": "string"},
        "provider_mode": {
            "type": "string",
            "enum": ["auto", "brave", "tavily", "both"],
        },
        "tool_mode": {
            "type": "string",
            "enum": ["auto", "safe", "deep", "manual"],
        },
        "freshness_sensitive": {"type": "boolean"},
    },
    "required": [
        "planned_queries",
        "coverage_map",
        "dedupe_notes",
        "provider_mode",
        "tool_mode",
    ],
}


FOLLOW_UP_PLANNER_SYSTEM_PROMPT = """You are a follow-up research planner.

Your job is to consolidate multiple user-approved follow-up questions into ONE minimal, non-overlapping research plan.

Requirements:
- Cover EVERY approved question in your coverage map
- Minimize query overlap — consolidate related questions into shared searches
- Reuse sources where possible (pricing, comparisons, etc.)
- Keep total planned queries between 2 and {max_queries} (hard limit)
- Respect provider_mode and depth constraints

Your output must explain:
1. planned_queries: The consolidated search queries to run
2. coverage_map: Which approved question each planned query answers
3. dedupe_notes: Brief explanation of consolidation strategy

Input context will include:
- approved_questions: The user's approved research directions
- source_context: Information about the original content
- summary: The existing summary text

Output format:
{{
  "planned_queries": ["query1", "query2", ...],
  "coverage_map": [
    {{"approved_question": "Q1", "covered_by": ["query1"]}},
    {{"approved_question": "Q2", "covered_by": ["query1", "query2"]}}
  ],
  "dedupe_notes": "Q1 and Q2 combined for pricing; Q3 needs separate search"
}}
""".format(max_queries=MAX_PLANNED_QUERIES)


def _clip_queries(queries: list[str]) -> list[str]:
    """Deduplicate and limit queries."""
    seen = set()
    result = []
    for q in queries:
        if q is None:
            continue
        normalized = q.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(q.strip())
    return result[:MAX_PLANNED_QUERIES]


def _normalize_coverage_map(
    approved_questions: list[str],
    planned_queries: list[str],
    coverage_map: list[dict] | None,
) -> list[dict]:
    """Ensure each approved question maps to valid planned queries."""
    valid_queries = set(planned_queries)
    fallback_queries = planned_queries or approved_questions
    normalized_entries: dict[str, list[str]] = {}

    for entry in coverage_map or []:
        if not isinstance(entry, dict):
            continue
        approved_question = str(entry.get("approved_question") or "").strip()
        if approved_question not in approved_questions or approved_question in normalized_entries:
            continue
        raw_covered_by = entry.get("covered_by")
        if not isinstance(raw_covered_by, list):
            raw_covered_by = []
        covered_by = _clip_queries([
            str(query).strip()
            for query in raw_covered_by
            if str(query).strip() in valid_queries
        ])
        normalized_entries[approved_question] = covered_by or list(fallback_queries)

    return [
        {
            "approved_question": question,
            "covered_by": normalized_entries.get(question, list(fallback_queries)),
        }
        for question in approved_questions
    ]


def plan_follow_up_research(
    *,
    source_context: dict,
    summary: str,
    approved_questions: list[str],
    question_provenance: list[QuestionProvenance] = None,
    provider_mode: str = DEFAULT_PROVIDER_MODE,
    tool_mode: str = DEFAULT_TOOL_MODE,
    depth: str = DEFAULT_DEPTH,
) -> FollowUpResearchPlan:
    """
    Generate consolidated research plan from approved user questions.

    The planner sees ALL approved questions together and generates ONE minimal
    non-overlapping query plan. This avoids redundant searches and ensures
    shared sources across all questions.

    Args:
        source_context: Information about the original content (url, type, etc.)
        summary: The existing summary text
        approved_questions: User-approved research directions (max 3)
        question_provenance: Where each question came from (suggested/preset/custom)
        provider_mode: Research provider mode
        tool_mode: Research tool mode
        depth: Research depth

    Returns:
        FollowUpResearchPlan with consolidated query plan and coverage map
    """
    if len(approved_questions) > MAX_APPROVED_QUESTIONS:
        raise ValueError(f"Maximum {MAX_APPROVED_QUESTIONS} questions allowed")

    # Clean inputs
    clean_questions = _clip_queries(approved_questions)
    if not clean_questions:
        raise ValueError("At least one approved question is required")

    clean_provider = provider_mode if provider_mode in {"auto", "brave", "tavily", "both"} else DEFAULT_PROVIDER_MODE
    clean_tool_mode = tool_mode if tool_mode in {"auto", "safe", "deep", "manual"} else DEFAULT_TOOL_MODE
    clean_depth = depth if depth in {"quick", "balanced", "deep"} else DEFAULT_DEPTH

    # Determine if this is freshness-sensitive
    published_at = source_context.get("published_at", "")
    freshness_sensitive = _looks_time_sensitive(published_at, summary, source_context)

    # Build context for planner
    user_context = [
        f"Original content: {source_context.get('title', 'Unknown')}",
        f"URL: {source_context.get('url', 'N/A')}",
        f"Published: {published_at or 'Unknown'}",
        f"Content type: {source_context.get('type', 'unknown')}",
    ]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": FOLLOW_UP_PLANNER_SYSTEM_PROMPT}
    ]

    # Add summary as context (truncated if very long)
    summary_context = summary[:2000] if len(summary) > 2000 else summary
    messages.append({
        "role": "user",
        "content": "\n".join([
            *user_context,
            "",
            "Existing Summary:",
            summary_context,
            "",
            f"Approved follow-up questions ({len(clean_questions)}):",
            *[f"- {q}" for q in clean_questions],
            "",
            "Generate a consolidated research plan.",
        ])
    })

    try:
        parsed, llm_provider, llm_model = chat_json_schema(
            messages=messages,
            schema_name="FollowUpResearchPlan",
            schema=FOLLOW_UP_PLAN_SCHEMA,
            max_tokens=PLANNER_MAX_TOKENS,
            reasoning_effort="low",
            temperature=0.0,
            timeout=PLANNER_TIMEOUT_SECONDS,
            provider=RESEARCH_PLANNER_PROVIDER,
        )

        planned_queries = _clip_queries(list(parsed.get("planned_queries") or []))
        if not planned_queries:
            # Fallback: use approved questions directly
            planned_queries = clean_questions

        coverage_map = _normalize_coverage_map(
            clean_questions,
            planned_queries,
            parsed.get("coverage_map", []),
        )
        dedupe_notes = str(parsed.get("dedupe_notes") or "Consolidated plan generated")

        planner_provider_mode = str(parsed.get("provider_mode", clean_provider)).strip().lower()
        if planner_provider_mode not in {"auto", "brave", "tavily", "both"}:
            planner_provider_mode = clean_provider

        planner_tool_mode = str(parsed.get("tool_mode", clean_tool_mode)).strip().lower()
        if planner_tool_mode not in {"auto", "safe", "deep", "manual"}:
            planner_tool_mode = clean_tool_mode

        # Detect comparison intent
        compare = _looks_like_comparison(clean_questions)

        return FollowUpResearchPlan(
            approved_questions=clean_questions,
            question_provenance=question_provenance or ["suggested"] * len(clean_questions),
            question_kinds=_infer_question_kinds(clean_questions, source_context),
            planned_queries=planned_queries,
            coverage_map=coverage_map,
            dedupe_notes=dedupe_notes,
            provider_mode=planner_provider_mode,  # type: ignore[arg-type]
            tool_mode=planner_tool_mode,  # type: ignore[arg-type]
            depth=clean_depth,  # type: ignore[arg-type]
            compare=compare,
            freshness_sensitive=freshness_sensitive,
            planner_provider=llm_provider,
            planner_model=llm_model,
        )

    except Exception as exc:
        logger.warning(f"Follow-up planning failed: {exc}, using fallback")
        # Fallback: use approved questions as planned queries
        return FollowUpResearchPlan(
            approved_questions=clean_questions,
            question_provenance=question_provenance or ["suggested"] * len(clean_questions),
            question_kinds=_infer_question_kinds(clean_questions, source_context),
            planned_queries=clean_questions,
            coverage_map=[
                {"approved_question": q, "covered_by": clean_questions}
                for q in clean_questions
            ],
            dedupe_notes="Fallback: using approved questions directly",
            provider_mode=clean_provider,  # type: ignore[arg-type]
            tool_mode=clean_tool_mode,  # type: ignore[arg-type]
            depth=clean_depth,  # type: ignore[arg-type]
            compare=_looks_like_comparison(clean_questions),
            freshness_sensitive=freshness_sensitive,
            planner_provider="fallback",
            planner_model="deterministic",
        )


def suggest_follow_up_questions(
    *,
    source_context: dict,
    summary: str,
    entities: list[str] = None,
    max_suggestions: int = 3,
) -> list[FollowUpSuggestion]:
    """
    Generate follow-up research suggestions using the planner LLM.

    The planner analyzes the summary and source context to determine
    if follow-up suggestions are appropriate, and generates contextual
    questions if so.

    Args:
        source_context: Information about the original content
        summary: The existing summary text
        entities: Optional list of extracted entities (for context)
        max_suggestions: Maximum number of suggestions to generate

    Returns:
        List of FollowUpSuggestion (may be empty if suggestions not appropriate)
    """
    # Extract entities from summary if not provided (for LLM context)
    if entities is None:
        entities = _extract_entities(summary)

    # Use planner to generate suggestions (LLM decides if appropriate)
    suggestions = _generate_suggestions_with_planner(
        source_context=source_context,
        summary=summary,
        entities=entities,
        max_suggestions=max_suggestions,
    )

    return suggestions


def _generate_suggestions_with_planner(
    *,
    source_context: dict,
    summary: str,
    entities: list[str],
    max_suggestions: int,
) -> list[FollowUpSuggestion]:
    """Use planner LLM to generate follow-up suggestions."""
    # Build context for planner
    user_context = [
        f"Title: {source_context.get('title', 'Unknown')}",
        f"URL: {source_context.get('url', 'N/A')}",
        f"Published: {source_context.get('published_at', 'Unknown')}",
        f"Type: {source_context.get('type', 'unknown')}",
    ]

    if entities:
        user_context.append(f"Identified entities: {', '.join(entities[:5])}")

    # Truncate summary if very long
    summary_context = summary[:3000] if len(summary) > 3000 else summary

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join([
                *user_context,
                "",
                "Summary:",
                summary_context,
                "",
                f"Generate up to {max_suggestions} follow-up research suggestions.",
            ])
        },
    ]

    try:
        parsed, llm_provider, llm_model = chat_json_schema(
            messages=messages,
            schema_name="FollowUpSuggestions",
            schema=SUGGESTIONS_SCHEMA,
            max_tokens=PLANNER_MAX_TOKENS,
            reasoning_effort="low",
            temperature=0.1,
            timeout=PLANNER_TIMEOUT_SECONDS,
            provider=RESEARCH_PLANNER_PROVIDER,
        )

        should_suggest = bool(parsed.get("should_suggest", False))
        if not should_suggest:
            logger.info("Planner determined follow-up suggestions are not appropriate: %s",
                       parsed.get("explanation", "No explanation provided"))
            return []

        raw_suggestions = parsed.get("suggestions") or []
        if not raw_suggestions:
            logger.warning("Planner returned should_suggest=true but no suggestions")
            return []

        # Convert raw suggestions to FollowUpSuggestion dataclass
        suggestions = []
        for i, s in enumerate(raw_suggestions):
            kind = s.get("kind", "background")
            # Validate kind is one of the allowed values
            allowed_kinds = ["current_state", "pricing", "comparison", "alternatives", "fact_check", "background", "what_changed"]
            if kind not in allowed_kinds:
                kind = "background"

            suggestions.append(FollowUpSuggestion(
                id=s.get("id", f"suggestion-{i+1}"),
                label=s.get("label", f"Follow-up Question {i+1}"),
                question=s.get("question", ""),
                reason=s.get("reason", ""),
                kind=kind,  # type: ignore[arg-type]
                priority=float(s.get("priority", 0.5)),
                default_selected=bool(s.get("default_selected", i == 0)),
                provenance="suggested",  # LLM-generated
            ))

        logger.info("Generated %d follow-up suggestions via %s (%s)",
                   len(suggestions), llm_provider, llm_model)
        return suggestions[:max_suggestions]

    except Exception as exc:
        logger.warning("Failed to generate LLM suggestions, using fallback: %s", exc)
        # Fallback to heuristic suggestions
        return _generate_fallback_suggestions(
            source_context=source_context,
            summary=summary,
            entities=entities,
            max_suggestions=max_suggestions,
        )


def _generate_fallback_suggestions(
    *,
    source_context: dict,
    summary: str,
    entities: list[str],
    max_suggestions: int,
) -> list[FollowUpSuggestion]:
    """Generate heuristic fallback suggestions when LLM fails."""
    suggestions = []
    published_at = source_context.get("published_at", "")
    is_old_content = _is_content_old(published_at)

    if is_old_content:
        suggestions.append(FollowUpSuggestion(
            id="what-changed-fallback",
            label="What changed since this was published?",
            question=f"What has changed since {published_at or 'publication'} regarding the topics in this content?",
            reason="Content may be outdated, recent updates may be available",
            kind="what_changed",
            priority=0.9,
            default_selected=True,
            provenance="preset",
        ))

    # Check for pricing/product mentions
    has_pricing = any(
        keyword in summary.lower()
        for keyword in ["price", "cost", "subscription", "$", "free tier", "paid"]
    )

    if has_pricing and entities:
        suggestions.append(FollowUpSuggestion(
            id="current-pricing-fallback",
            label="What is the current pricing?",
            question=f"What is the current pricing and availability for {entities[0]}?",
            reason="Pricing information changes frequently",
            kind="pricing",
            priority=0.8,
            default_selected=False,
            provenance="preset",
        ))

    # Check for comparison opportunity
    if len(entities) >= 2:
        suggestions.append(FollowUpSuggestion(
            id="comparison-fallback",
            label="How do these compare to alternatives?",
            question=f"How does {entities[0]} compare to {entities[1]} and alternatives today?",
            reason="Multiple products/entities mentioned, comparison may be useful",
            kind="comparison",
            priority=0.7,
            default_selected=False,
            provenance="preset",
        ))

    return suggestions[:max_suggestions]


# Helper functions

def _looks_time_sensitive(published_at: str, summary: str, source_context: dict) -> bool:
    """Check if content is time-sensitive and may have updates."""
    # Check if content is old (> 3 months)
    if published_at:
        is_old = _is_content_old(published_at)
        if is_old:
            return True

    # Check for time-sensitive keywords
    time_sensitive_keywords = [
        "pricing", "cost", "subscription", "free tier", "paid",
        "latest", "newest", "current", "recent", "2024", "2025",
        "version", "release", "announcement",
    ]

    summary_lower = summary.lower()
    return any(keyword in summary_lower for keyword in time_sensitive_keywords)


def _is_content_old(published_at: str) -> bool:
    """Check if content is older than 3 months."""
    try:
        from datetime import datetime, timedelta
        pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return datetime.now(pub_date.tzinfo) - pub_date > timedelta(days=90)
    except Exception:
        return False


def _looks_like_comparison(questions: list[str]) -> bool:
    """Detect if approved questions are asking for comparison."""
    comparison_keywords = [
        "compare", "vs", "versus", "difference", "alternative",
        "better than", "worse than", "similar to"
    ]

    questions_text = " ".join(questions).lower()
    return any(keyword in questions_text for keyword in comparison_keywords)


def _infer_question_kinds(questions: list[str], source_context: dict) -> list[QuestionKind]:
    """Infer question kind from question text and context."""
    kinds = []

    for question in questions:
        lower_q = question.lower()

        if any(kw in lower_q for kw in ["price", "cost", "subscription", "tier"]):
            kinds.append("pricing")
        elif any(kw in lower_q for kw in ["compare", "vs", "versus", "alternative"]):
            kinds.append("comparison")
        elif any(kw in lower_q for kw in ["changed", "since publication", "since release", "since this was published", "what changed"]):
            kinds.append("what_changed")
        elif any(kw in lower_q for kw in ["current", "currently", "state", "status", "today", "availability", "available", "right now"]):
            kinds.append("current_state")
        elif any(kw in lower_q for kw in ["update", "updated", "new since"]):
            kinds.append("what_changed")
        elif any(kw in lower_q for kw in ["true", "false", "fact check", "accurate"]):
            kinds.append("fact_check")
        else:
            kinds.append("background")

    return kinds


def _should_suggest_follow_up(source_context: dict, summary: str, entities: list[str] | None) -> bool:
    """Determine if follow-up suggestions are appropriate for this content."""
    # Don't suggest if no entities found
    if not entities:
        entities = _extract_entities(summary)
        if not entities:
            return False

    # Don't suggest for very recent content (< 1 week)
    published_at = source_context.get("published_at", "")
    if published_at and not _is_content_old(published_at):
        # Check if it's still worth suggesting (pricing, etc.)
        summary_lower = summary.lower()
        if not any(kw in summary_lower for kw in ["price", "$", "subscription", "compare"]):
            return False

    return True


def _extract_entities(text: str) -> list[str]:
    """Extract named entities from text (simple heuristic-based extraction)."""
    # This is a placeholder - could use NLP in production
    # For now, look for capitalised product/company names
    import re

    # Look for patterns like "GPT-5", "Cursor AI", "Windsurf"
    patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Multi-word proper nouns
        r"\bGPT-\d+\b",  # GPT models
        r"\b[A-Z]+-\d+\b",  # Product-version patterns
    ]

    entities = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # Filter out common words
    common_words = {"The", "This", "That", "These", "Those", "A", "An"}
    entities = [e for e in entities if e not in common_words and len(e) > 2]

    return sorted(list(entities))[:10]  # Max 10 entities


def build_cache_key(
    video_id: str,
    summary_id: Optional[int],
    approved_questions: list[str],
    provider_mode: str,
    depth: str,
) -> str:
    """
    Build cache key for follow-up research results.

    Matches the schema used in follow_up_research_runs table.
    """
    return _build_cache_key(video_id, summary_id, approved_questions, provider_mode, depth)

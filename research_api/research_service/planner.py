"""Research planning step (query generation + provider/tool strategy)."""

from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_DEPTH,
    DEFAULT_PROVIDER_MODE,
    DEFAULT_TOOL_MODE,
    DEPTH_QUERY_LIMIT,
    MAX_HISTORY_TURNS,
    PLANNER_MAX_TOKENS,
    PLANNER_TIMEOUT_SECONDS,
    RESEARCH_PLANNER_PROVIDER,
)
from .llm import chat_json_schema
from .models import ResearchPlan

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "provider_mode": {"type": "string", "enum": ["auto", "brave", "tavily", "both"]},
        "tool_mode": {"type": "string", "enum": ["auto", "safe", "deep", "manual"]},
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "comparison_axes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "freshness_sensitive": {"type": "boolean"},
        "requires_source_backed_answer": {"type": "boolean"},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
    },
    "required": [
        "objective",
        "queries",
        "provider_mode",
        "tool_mode",
        "entities",
        "comparison_axes",
        "freshness_sensitive",
        "requires_source_backed_answer",
        "needs_clarification",
    ],
    "additionalProperties": False,
}

PLANNER_SYSTEM_PROMPT = """You are a research planner.
Generate concrete web-search queries from user intent and recent context.

Guidelines:
- Queries should be specific and web-search ready.
- Prefer 1-3 focused queries unless depth is deep.
- For compare requests, identify the main entities and 2-5 useful comparison axes.
- Set freshness_sensitive=true for requests involving latest/current/recent/news/reviews/status/pricing/releases or anything likely to change over time.
- Set requires_source_backed_answer=true when the user explicitly asks to research, search the web, compare with sources, cite sources, or summarize what the web says.
- If the user request is under-specified for factual research, set needs_clarification=true and ask one direct question.
- provider_mode/tool_mode should usually respect UI preferences unless clearly better otherwise.
"""


def _clip_queries(queries: list[str], depth: str) -> list[str]:
    max_q = DEPTH_QUERY_LIMIT.get(depth, DEPTH_QUERY_LIMIT[DEFAULT_DEPTH])
    out_raw = [q.strip() for q in queries if q and q.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for q in out_raw:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:max_q] if out else []


def _clip_labels(values: list[str], max_items: int) -> list[str]:
    out_raw = [str(v).strip() for v in values if str(v).strip()]
    seen: set[str] = set()
    out: list[str] = []
    for value in out_raw:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def plan_research(
    *,
    message: str,
    history: list[dict[str, str]] | None,
    provider_mode: str,
    tool_mode: str,
    depth: str,
) -> ResearchPlan:
    clean_provider = provider_mode if provider_mode in {"auto", "brave", "tavily", "both"} else DEFAULT_PROVIDER_MODE
    clean_tool_mode = tool_mode if tool_mode in {"auto", "safe", "deep", "manual"} else DEFAULT_TOOL_MODE
    clean_depth = depth if depth in {"quick", "balanced", "deep"} else DEFAULT_DEPTH

    turns = (history or [])[-MAX_HISTORY_TURNS:]

    user_context = [
        f"UI provider_mode preference: {clean_provider}",
        f"UI tool_mode preference: {clean_tool_mode}",
        f"UI depth: {clean_depth}",
        "Return provider_mode/tool_mode compatible with these preferences unless user intent strongly requires change.",
    ]

    messages: list[dict[str, str]] = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
    for turn in turns:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": "\n".join(user_context + [f"Research request: {message}"])})

    try:
        parsed, llm_provider, llm_model = chat_json_schema(
            messages=messages,
            schema_name="ResearchPlan",
            schema=PLAN_SCHEMA,
            max_tokens=PLANNER_MAX_TOKENS,
            reasoning_effort="low",
            temperature=0.0,
            timeout=PLANNER_TIMEOUT_SECONDS,
            provider=RESEARCH_PLANNER_PROVIDER,
        )

        queries = _clip_queries(list(parsed.get("queries") or []), clean_depth)
        if not queries:
            queries = [message.strip()]

        planner_provider = str(parsed.get("provider_mode", clean_provider)).strip().lower()
        if planner_provider not in {"auto", "brave", "tavily", "both"}:
            planner_provider = clean_provider

        planner_tool_mode = str(parsed.get("tool_mode", clean_tool_mode)).strip().lower()
        if planner_tool_mode not in {"auto", "safe", "deep", "manual"}:
            planner_tool_mode = clean_tool_mode

        needs_clarification = bool(parsed.get("needs_clarification", False))
        clarification_question = str(parsed.get("clarification_question", "") or "").strip()

        if needs_clarification and not clarification_question:
            needs_clarification = False

        return ResearchPlan(
            objective=str(parsed.get("objective", message) or message),
            queries=queries,
            provider_mode=planner_provider,  # type: ignore[arg-type]
            tool_mode=planner_tool_mode,  # type: ignore[arg-type]
            llm_provider=llm_provider,
            llm_model=llm_model,
            entities=_clip_labels(list(parsed.get("entities") or []), 6),
            comparison_axes=_clip_labels(list(parsed.get("comparison_axes") or []), 6),
            freshness_sensitive=bool(parsed.get("freshness_sensitive", False)),
            requires_source_backed_answer=bool(parsed.get("requires_source_backed_answer", False)),
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
        )
    except Exception:
        return ResearchPlan(
            objective=message.strip() or "research request",
            queries=[message.strip() or "latest updates"],
            provider_mode=clean_provider,  # type: ignore[arg-type]
            tool_mode=clean_tool_mode,  # type: ignore[arg-type]
            llm_provider="fallback",
            llm_model="deterministic",
            entities=[],
            comparison_axes=[],
            freshness_sensitive=False,
            requires_source_backed_answer=False,
            needs_clarification=False,
            clarification_question="",
        )
